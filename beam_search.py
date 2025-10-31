from models.Transformer import make_model
from models.Transformer_modules import DataGenerator
from utils.utils import load_net_state, seed_everything, top_k_eval, eval_canonical_smiles, get_seq
import torch, json
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
import torch
import torch.nn.functional as F
from typing import List
from utils.dataloader import make_testloader
from tqdm import tqdm
import numpy as np


seed_everything(2024)


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
                generator,
                beam_size: int = 4,
                max_len: int = 50,
                bos=0,
                eos: int = 2,
                length_penalty: float = 0,
                temperature=15,
                repetition_penalty=1,
                device="cpu",
                stochastic=1):
    # fbank_feat: the fbank feature of input audio. (b, t, dim)
    # feat_lens: the lengths of fbank features.  (b,)

    # bs: batch size, beam_size: beam size
    if type(model) == torch.nn.parallel.DistributedDataParallel:
        model = model.module 
    batch_size = generator.src.shape[0]    # rns: running size, equal to batch size * beam size
    rns = batch_size * beam_size

    # init hypotheses, scores and flags
    hyps = torch.tensor([[bos]]).long().repeat(batch_size, 1)  # (bs, 1)
    hyps = hyps.unsqueeze(1).repeat(1, beam_size, 1).view(rns, 1)  # (rns, 1), the hypothesis of current beam
    scores = torch.zeros(beam_size).float()
    scores[1:] = float("-inf")
    scores = scores.repeat(batch_size, 1).view(rns)                     # (rns,), the scores of current beam
    end_flag = torch.zeros(rns).bool()                         # (rns,), whether current beam is finished
    
    hyps = hyps.to(generator.device)
    scores = scores.to(generator.device)
    end_flag = end_flag.to(generator.device)

    # get encoder output (memory)
    memory = model.encode(generator.src, None)
    # feat_lens = torch.tensor([memory.shape[-1]], device=generator.device)
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
                              torch.ones_like(hyps, device=generator.device))
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
        
    # scores = scores.view(-1, beam_size)  # (rns, beam_size)
    # _, best_hyp_idxs = scores.topk(k=beam_size, dim=-1)  # (bs, 1)
    # best_hyp_idxs = best_hyp_idxs.view(-1)
    idxs = torch.arange(batch_size+1, device=scores.device) * beam_size
    idxs = idxs.unsqueeze(1).repeat(1, 1).view(-1)
    # best_hyp_idxs += idxs
    # best_hyps = torch.index_select(hyps, dim=0, index=best_hyp_idxs)

    # pred_tokens = best_hyps[:, 1:]
    # pred_tokens = [hyp[hyp!= eos].tolist() for hyp in pred_tokens]

    return [hyps[:, 1:][idxs[i]:idxs[i+1], :] for i in range(len(idxs)-1)]


vocab_size = 181
model, src_length = make_model(vocab_size, N=4, d_model=512)
checkpoint = 'checkpoints/raman2mol/Transformer/2025-09-28_15_18/490_acc_0148.pth'#'checkpoints/raman2mol/Transformer/2025-09-16_20_19/1110_acc_1705.pth'
ds = checkpoint.split('/')[1]
device = torch.device('cuda:3')
model = load_net_state(model, torch.load(checkpoint, map_location=device, weights_only=True)['model_state']).to(device)
# data = pd.read_pickle('datasets/raman2mol/test.pkl')
# x, y = data['spectrum'], data['smiles']
# dataset = MyDataset(x, y)
loader = make_testloader(ds,2)
lens = []
temps = [1]
beamsizes = [3]
recalls = []
for temperature in temps:
    for beam_size in beamsizes:
        top_k_result = np.zeros(beam_size)
        # result = []
        for _, batch in enumerate(tqdm(loader, ncols=50, total=len(loader))):
            data_iter = DataGenerator(batch, src_length, device)
            # pred = greedy_decode(model, 
            #               data_iter,
            #               src_length,
            #               max_len=30)
            # result.append(eval_canonical_smiles([get_seq(o) for o in pred], 
            #                                     [get_seq(t) for t in data_iter.true_seq]))
            lens.append(len(data_iter.src))
            prediction = beam_search(model, 
                            data_iter,
                            device=device,
                            beam_size=beam_size,
                            max_len=30,
                            length_penalty=0,
                            temperature=temperature,
                            stochastic=0)
            # result.append(prediction)

        # for pred_batch, target_batch in zip(result, loader):
            for pred, target in zip(prediction, data_iter.tgt_y):
                top_k_result += top_k_eval(pred, target, beam_size)
            # for p in range(len(pred)):
                # print(f'{p+1}: Prediction: {get_seq(pred[p])}, Target: {get_seq(target)}')
        recalls.append(sum(top_k_result)/sum(lens))
        print(f'beams={beam_size}, temp={temperature}, topk recall={sum(top_k_result)/sum(lens)}')
print(top_k_result)
print(recalls)

# greedy_decode
# num_correct =  0
# truelist, falselist = [], []
# for pred_batch, target_batch in zip(result, loader):
#     for pred, target in zip(pred_batch, DataGenerator(target_batch, src_length, device).tgt_y):
#             p, t = get_seq(pred), get_seq(target)
#             print(f'Prediction: {p}, Target: {t}')
#             num_correct += p==t
#             if p==t: truelist.append([p,t])
#             else: falselist.append([p,t])
# print(num_correct)
# pd.to_pickle({'true':truelist, 'false':falselist},'search.pkl')
# for i in result:
#     for smiles in i:print(smiles)

