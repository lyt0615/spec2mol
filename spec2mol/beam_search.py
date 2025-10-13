from models.Transformer import make_model
from utils.utils import load_net_state, seed_everything, get_smiles
import torch, json
import pandas as pd
import torch
import torch.nn.functional as F
from typing import List
import numpy as np


seed_everything(2024)


def get_smiles(label):
    with open("models/moltokenizer/vocab.json",'r',encoding='utf-8') as f:
        tokens = {val: key for key, val in json.load(f).items()}
    smiles = ''
    if type(label) == torch.Tensor: label = label.cpu().detach() 
    for l in label:
        smiles += tokens[l.data.item()]
    smiles = smiles.replace('</s>', '')
    smiles = smiles.replace('<s>', '')
    smiles = smiles.replace('<unk>', '')
    smiles = smiles.replace('<pad>', '')
    return smiles


class BeamHypothesis:
    """维护一条候选序列：tokens + log_prob + kv_cache"""
    __slots__ = ["tokens", "score", "past"]

    def __init__(self, tokens: List[int], score: float, past=None):
        self.tokens = tokens
        self.score  = score
        self.past   = past

    def __len__(self):
        return len(self.tokens)


def gumbel_topk(logits, k, eps=1e-20):
    """logits: (V,)   返回 (values, indices)  与 torch.topk 接口一致"""
    U = torch.rand(logits.shape, device=logits.device)
    gumbel_logits = logits - torch.log(-torch.log(U + eps) + eps)
    return torch.topk(gumbel_logits, k)


def mask_finished_scores(scores, end_flag, inf=-float("inf")):
    rns, beam_size = scores.size()
    assert end_flag.size(0) == rns and end_flag.ndim == 1
    zero_mask = scores.new_zeros(rns, 1)
    mask_to_zero = torch.cat([end_flag.view(rns, 1), zero_mask.repeat(1, beam_size - 1)], dim=-1)  # (rns, beam_size)
    mask_to_inf = torch.cat([zero_mask, end_flag.view(rns, 1).repeat(1, beam_size - 1)], dim=-1)  # (rns, beam_size)
    scores = scores.masked_fill(mask_to_zero.bool(), 0.)
    scores = scores.masked_fill(mask_to_inf.bool(), inf)
    return scores

def mask_finished_preds(preds, end_flag, eos):
    # Force preds to be all `sos` for finished beams.
    _, beam_size = preds.size()
    finished = end_flag.view(-1, 1).repeat(1, beam_size)  # (rns, beam_size)
    preds.masked_fill_(finished.bool(), eos)
    return preds


@torch.no_grad()
def beam_search(model,
                spec,
                beam_size: int = 4,
                max_len: int = 50,
                device=torch.device('cpu'),
                bos=0,
                eos: int = 2,
                length_penalty: float = 0,
                temperature=15,
                repetition_penalty=1,
                stochastic=1):
    # fbank_feat: the fbank feature of input audio. (b, t, dim)
    # feat_lens: the lengths of fbank features.  (b,)

    # bs: batch size, beam_size: beam size
    if type(model) == torch.nn.parallel.DistributedDataParallel:
        model = model.module 
    batch_size = 1
    # predlist = []   
    rns = beam_size * batch_size # rns: running size, equal to batch size * beam size

    # init hypotheses, scores and flags
    hyps = torch.tensor([[bos]]).long().repeat(batch_size, 1)  # (bs, 1)
    hyps = hyps.unsqueeze(1).repeat(1, beam_size, 1).view(rns, 1)  # (rns, 1), the hypothesis of current beam
    scores = torch.zeros(beam_size).float()
    scores[1:] = float("-inf")
    scores = scores.repeat(batch_size, 1).view(rns)                     # (rns,), the scores of current beam
    end_flag = torch.zeros(rns).bool()                         # (rns,), whether current beam is finished
    
    hyps = hyps.to(device)
    scores = scores.to(device)
    end_flag = end_flag.to(device)
    
    spec = torch.FloatTensor(spec[::-1].copy())
    spec = spec.reshape(1, 1, spec.shape[-1]) if spec.dim() != 3 else spec
    # get encoder output (memory)
    memory = model.encode(spec, None)
    # feat_lens = torch.tensor([memory.shape[-1]], device=device)
    memory = memory.unsqueeze(1).repeat(1, beam_size, 1, 1).view(rns, memory.size(-2), memory.size(-1))
    # feat_lens = feat_lens.unsqueeze(1).repeat(1, beam_size).view(rns,)

    # main loop
    for i in range(max_len):

        # check whether all beams are finished
        if end_flag.all():
            break

        # forward
        # dec_mask = model.get_subsequent_mask(rns, hyps.size(1), hyps.device)
        # dec_enc_mask = model.get_enc_dec_mask(rns, memory.size(1), feat_lens, hyps.size(1), hyps.device)
        output = model.decode(memory, 
                              None,
                              hyps,
                              torch.ones_like(hyps, device=device))
        logits = model.generator(output[0]) if type(output) == tuple else output
        logits = logits[:, -1] / temperature
        logp = F.softmax(logits, dim=-1)  # (rns, vocab)
        # local pruning: prune non-topk scores
        topk_logp, topk_idxs = gumbel_topk(logp, beam_size) if stochastic else logp.topk(beam_size, -1) # (rns, vocab) -> (rns, beam_size)

        # masked finished beams
        topk_logp = mask_finished_scores(topk_logp, end_flag)
        topk_idxs = mask_finished_preds(topk_idxs, end_flag, eos)

        # calculate scores of new beams
        scores = scores.view(rns, 1)
        scores = scores + topk_logp  # (rns, 1) + (rns, beam_size) -> (rns, beam_size)
        scores = scores.view(batch_size, beam_size * beam_size)
            
        # global pruning
        scores, offset_k_idxs = scores.topk(beam_size, -1)  # (bs, beam_size)
        scores = scores.view(rns, 1)
        offset_k_idxs = offset_k_idxs.view(-1)

        # .1: calculate the predicted token at current decoding step
        base_k_idxs = torch.arange(batch_size, device=scores.device) * beam_size * beam_size
        # e.g. base_k_idxs: (0, 0, 0, 9, 9, 9, 27, 27, 27, 36, 36, 36)
        base_k_idxs = base_k_idxs.unsqueeze(-1).repeat(1, beam_size).view(-1)
        best_k_idxs = base_k_idxs + offset_k_idxs
        best_k_pred = torch.index_select(topk_idxs.view(-1), dim=-1, index=best_k_idxs)

        # .2: retrive the old hypotheses of best k beams
        best_hyp_idxs = best_k_idxs.div(beam_size, rounding_mode="floor")
        last_best_k_hyps = torch.index_select(hyps, dim=0, index=best_hyp_idxs)  # (rns, i)

        # .3: concat the old hypotheses with the new predicted token
        hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)), dim=1)  # (rns, i)

        # refresh end_flag
        end_flag = torch.eq(hyps[:, -1], eos).view(-1)
        live_scores = [scores[i].data.item()/(hyps.shape[-1]**length_penalty) for i in range(len(scores)) if hyps[i][-1]!=eos]
        done_scores = [scores[i].data.item()/(hyps.shape[-1]**length_penalty) for i in range(len(scores)) if hyps[i][-1]==eos]
        if live_scores and done_scores and max(live_scores) < max(done_scores):
            break 
        # length penalty
        # new_beam.sort(key=lambda x: x.score / (len(x) ** length_penalty), reverse=True)
        # live_scores = [hyp.score / (len(hyp.tokens)**length_penalty) for hyp in new_beam if hyp.tokens[-1] != eos]
        # done_scores = [hyp.score / (len(hyp.tokens)**length_penalty) for hyp in new_beam if hyp.tokens[-1] == eos]
        # if live_scores and done_scores and max(live_scores) < max(done_scores):
        #     beams = next_beams
        #     break  
    
    idxs = torch.arange(batch_size+1, device=scores.device) * beam_size
    idxs = idxs.unsqueeze(1).repeat(1, 1).view(-1)
    pred_seq = [hyps[:, 1:][idxs[i]:idxs[i+1], :] for i in range(len(idxs)-1)]
    prediction = [get_smiles(s) for s in pred_seq[0]]
    return prediction

if __name__ == "__main__":
    spec = pd.read_pickle('/home/lyt/projects/spec2mol/datasets/ir2mol/test.pkl')['spectrum'].values[0]#torch.randn(1024) ### input spectrum: 长度1024，波数范围400-4000
    checkpoint = 'spec2mol/checkpoints/checkpoint.pth'
    device = torch.device('cpu')
    vocab_size = 181
    model, src_length = make_model(vocab_size, N=4, d_model=512)
    model = load_net_state(model, torch.load(checkpoint, map_location=device, weights_only=True)['model_state']).to(device)
    temperature = 1
    beam_size = 3  # 返回的候选结构数

    top_k_result = np.zeros(beam_size)
    prediction = beam_search(model, 
                    spec,
                    beam_size=beam_size,
                    device=device,
                    max_len=30,
                    length_penalty=0,
                    temperature=temperature,
                    stochastic=0)
    print(f'Top {beam_size} predictions: {prediction}')

