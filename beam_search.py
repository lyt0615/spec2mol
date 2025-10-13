from models.Transformer import make_model
from models.Transformer_modules import DataGenerator, greedy_decode
from utils.utils import load_net_state, seed_everything, top_k_eval, eval_canonical_smiles, get_smiles
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

def beam_search_(model,
                generator,
                vocab_size: int,
                beam_size: int = 4,
                max_len: int = 50,
                pad: int = 1,
                bos=0,
                eos: int = 2,
                length_penalty: float = 0,
                temperature=1,
                repetition_penalty=2,
                device="cpu") -> List[List[int]]:
    """
    自回归束搜索
    :param model: 可调用，返回 (logits, past)
    :param prompt_ids: 初始 prompt 的 token ids
    :return: 按 score 降序排列的 top-k 序列
    """
    # 初始化

    batch_size = len(generator.src)
    src_channel, src_length = generator.src.shape[1], generator.src.shape[2]
    # prompt_ids = [[bos]] * batch_size
    # max_prompt = max(len(p) for p in prompt_ids)

    # 1) 统一 prompt 长度，做成 tensor
    prompt_tensor = torch.ones(batch_size, 1, device=generator.device).fill_(bos).type_as(generator.tgt)
    # torch.full((batch_size, max_prompt), tokenizer.pad_token_id,
    #                            dtype=torch.long, device=device)
    # for i, p in enumerate(prompt_ids):
    #     prompt_tensor[i, :len(p)] = torch.as_tensor(p, device=device)
    # 
    # input_ids = torch.ones(batch_size, 1, device=generator.device).fill_(0).type_as(generator.tgt)
    # prompt_ids = input_ids.tolist()
    logits, past = model.forward(generator.src, 
                         prompt_tensor,
                         None,
                         torch.ones_like(prompt_tensor, device=generator.device),
                         )
    logits = logits[:, -1, :]  # 取最后一个位置 [vocab]
            # tgt_mask = torch.ones_like(ys, device=ys.device) #Variable(subsequent_mask(step+1).type_as(data_generator.tgt_mask))
            # out = model.decode(memory, src_mask,
            #                 Variable(ys),
            #                 Variable(tgt_mask))
    # 初始 top-k
    probs = F.softmax(logits, dim=-1)
    topk_p, topk_ids = torch.topk(probs, beam_size, dim=-1)  # [beam]

    # beams = [BeamHypothesis(prompt_ids + [idx.item()], score.item(), past)
    #          for idx, score in zip(topk_ids, topk_p)]
    # beams = [BeamHypothesis(prompt_ids + [idx], score, past)
    #          for idx, score in zip(topk_ids.tolist(), topk_p.tolist())]
    beams = [[BeamHypothesis([idx.item()], score.item(), None)
              for idx, score in zip(topk_ids[b], topk_p[b])]
             for b in range(batch_size)]

    past = [(k.unsqueeze(1).expand(-1, beam_size, -1, -1, -1)
                    .reshape(batch_size * beam_size, *k.shape[1:]),
                v.unsqueeze(1).expand(-1, beam_size, -1, -1, -1)
                    .reshape(batch_size * beam_size, *v.shape[1:]))
            for (k, v) in past] 
    # 4) 把 KV-cache 从 (B, ...) 扩成 (B*K, ...)
    # past = [(k.unsZ(k, v) in past]
     
    for step in range(1, max_len + 1):
        # 5-1) 收集每条 beam 的最新 token 与是否已结束
        flat_tokens = []          # 长度 = B*K
        done_mask = []            # True 表示该 beam 已生成 eos
        for b in range(batch_size):
            for k in range(beam_size):
                hyp = beams[b][k]
                flat_tokens.append(hyp.tokens[-1])
                done_mask.append(hyp.tokens[-1] == eos)

        # sample_size = batch_size*beam_size**step 
        flat_tokens = torch.as_tensor(flat_tokens, device=device).unsqueeze(1)  # (B*K, 1)
        # generator.src = generator.src.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(beam_size*batch_size, src_channel, src_length)
        # 5-2) 一步 forward（batch_size*beam_size 一起算）
       
        logits, past = model.forward(generator.src.unsqueeze(1)
                                     .expand(-1, beam_size, -1, -1)
                                     .reshape(beam_size*batch_size, src_channel, src_length), 
                         flat_tokens,
                         None,
                         torch.ones_like(flat_tokens, device=generator.device),
                         past_key_values=past
                         )
        logits = logits[:, -1, :] / temperature                  # (B*K, V)
        for b in range(batch_size):
            for k in range(beam_size):
                idx = b * beam_size + k
                seen = set(beams[b][k].tokens)
                for tok in seen:
                    logits[idx, tok] /= repetition_penalty
        probs = F.softmax(logits, dim=-1)

        # 5-3) 每条样本内部再 top-K
        next_beams = [[]] * batch_size
        for b in range(batch_size):
            start, end = b * beam_size, (b + 1) * beam_size
            scores = probs[start:end]            # (K, V)
            # 加上历史得分
            hist_score = torch.tensor([beams[b][k].score for k in range(beam_size)],
                                      device=device)  # (K,)
            total = hist_score.unsqueeze(1) + scores   # (K, V)
            flat = total.view(-1)                      # (K*V,)
            topk_p, topk_flat = torch.topk(flat, beam_size) 
            beam_idx = topk_flat // vocab_size          # 来自哪条旧 beam
            tok_idx  = topk_flat %  vocab_size          # 新 token
            # past = [(torch.index_select(k, 0, beam_idx), torch.index_select(v, 0, beam_idx)) for k, v in past]
            # for past_layer in past:
                # past_layer[0] = past_layer[0][]
            for rank in range(beam_size):
                old_i = beam_idx[rank].item()
                new_tok = tok_idx[rank].item()
                new_score = topk_p[rank].item()
                old_hyp = beams[b][old_i]
                # 已结束的 beam 直接复制，不再扩展
                if old_hyp.tokens[-1] == eos:
                    next_beams[b].append(old_hyp)
                else:
                    next_beams[b].append(
                        BeamHypothesis(old_hyp.tokens + [new_tok],
                                       new_score,)) # 这里可再细化存 KV
                # if b==0:
                #     print(beam_idx, tok_idx, old_i, new_tok)
            # 按得分排序
            next_beams[b].sort(key=lambda x: x.score / (len(x) ** length_penalty), reverse=True)
            next_beams[b] = next_beams[b][:beam_size]

        beams = next_beams
        # past = [(k[beam_idx], v[beam_idx]) for k, v in past]
        
        # print(f'Step{step}')
        # for beam in beams[0]:
        #     print(beam.tokens)
            
    # 6) 去掉 prompt 返回
    return [[hyp.tokens for hyp in b] for b in beams]


def beam_search1(model,
    generator,
    vocab_size: int,
    beam_size: int = 4,
    max_len: int = 50,
    pad: int = 1,
    bos=0,
    eos: int = 2,
    length_penalty: float = 0.6,
    temperature=1,
    repetition_penalty=1.1,
    device="cpu"):
    batch_size = generator.src.shape[0]
    
    src_encode = model.encode(generator.src, None)
    for i in range(max_len):

        # check whether all beams are finished
        if all([h.finished() for h in hyps]):
            break

        # iterate over all beams
        new_hyps = []
        for h in hyps:

            # forward
            l = torch.tensor(h.tokens, device=l.device).view(1, -1)
            # dec_mask = model.get_subsequent_mask(batch_size, l.size(1), l.device)
            # dec_enc_mask = model.get_enc_dec_mask(batch_size, enc_out.size(1), feat_lens, l.size(1), l.device)
            logits, past = model.forward(generator.src, 
                                prompt_tensor,
                                None,
                                torch.ones_like(prompt_tensor, device=generator.device),
                                )
            logits
            logits = model.get_logits(enc_out, l, dec_mask, dec_enc_mask)
            logits = logits[:, -1]          # (1, T, vocab) -> (1, vocab)
            p = F.softmax(logits, dim=-1)

            # local pruning: prune non-topk scores
            topk_p, topk_idxs = p.topk(k=beam_size, dim=-1)  # (1, vocab) -> (1, beam_size)
            topk_p, topk_idxs = topk_p.view(-1), topk_idxs.view(-1)   # (beam_size,), (beam_size,)

            # masked finished beams
            if h.finished():
                topk_p[0] = 0.
                topk_p[1:] = float("-inf")
                topk_idxs.fill_(eos)

            # calculate scores of new beams
            for j in range(beam_size):
                new_hyps.append(
                    Hypothesis.build_from_prev(h, topk_idxs[j].item(), topk_p[j].item())
                )

        # global pruning
        new_hyps = sorted(new_hyps, key=lambda x: x.score, reverse=True)
        hyps = new_hyps[:beam_size]


def gumbel_topk(logits, k, eps=1e-20):
    """logits: (V,)   返回 (values, indices)  与 torch.topk 接口一致"""
    U = torch.rand(logits.shape, device=logits.device)
    gumbel_logits = logits - torch.log(-torch.log(U + eps) + eps)
    return torch.topk(gumbel_logits, k)


@torch.no_grad()
def beam_search2(model,
                generator,
                beam_size: int = 4,
                max_len: int = 50,
                bos=0,
                eos: int = 2,
                length_penalty: float = 0,
                temperature=15,
                repetition_penalty=1,
                device="cpu",
                stochastic=1) -> List[List[int]]:
    """
    自回归束搜索
    :param model: 可调用，返回 (logits, past)
    :param prompt_ids: 初始 prompt 的 token ids
    :return: 按 score 降序排列的 top-k 序列
    """
    # 初始化

    batch_size = len(generator.src)
    src_channel, src_length = generator.src.shape[1], generator.src.shape[2]

    # 1) 统一 prompt 长度，做成 tensor
    prompt_tensor = torch.ones(batch_size, 1, device=generator.device).fill_(bos).type_as(generator.tgt)

    logits, past = model.forward(generator.src, 
                         prompt_tensor,
                         None,
                         torch.ones_like(prompt_tensor, device=generator.device),
                         )
    logits = logits[:, -1, :] / temperature  # 取最后一个位置 [vocab]

    # 初始 top-k
    probs = F.softmax(logits, dim=-1)
    topk_p, topk_ids = torch.topk(probs, beam_size, dim=-1) if stochastic else gumbel_topk(probs, beam_size) # [beam]

    beams = [[BeamHypothesis([idx.item()], score.item(), None)
              for idx, score in zip(topk_ids[b], topk_p[b])]
             for b in range(batch_size)]

    past = [(k.unsqueeze(1).expand(-1, beam_size, -1, -1, -1)
                    .reshape(batch_size * beam_size, *k.shape[1:]),
                v.unsqueeze(1).expand(-1, beam_size, -1, -1, -1)
                    .reshape(batch_size * beam_size, *v.shape[1:]))
            for (k, v) in past] 

    for step in range(max_len):
        # 5-1) 收集每条 beam 的最新 token 与是否已结束
        flat_tokens = []          # 长度 = B*K
        done_mask = []            # True 表示该 beam 已生成 eos
        for b in range(batch_size):
            for k in range(beam_size):
                hyp = beams[b][k]
                flat_tokens.append(hyp.tokens[-1])
                done_mask.append(hyp.tokens[-1] == eos)

        flat_tokens = torch.as_tensor(flat_tokens, device=device).unsqueeze(1)  # (B*K, 1)
       
        logits, past = model.forward(generator.src.unsqueeze(1)
                                     .expand(-1, beam_size, -1, -1) 
                                     .reshape(beam_size*batch_size, src_channel, src_length), 
                         flat_tokens,
                         None,
                         torch.ones_like(flat_tokens, device=generator.device),
                         past_key_values=past
                         )
        logits = logits[:, -1, :] / temperature                  # (B*K, V)
        for b in range(batch_size):
            for k in range(beam_size):
                idx = b * beam_size + k
                seen = set(beams[b][k].tokens)
                for tok in seen:
                    logits[idx, tok] /= repetition_penalty
        probs = F.softmax(logits, dim=-1)

        # 5-3) 每条样本内部再 top-K
        next_beams = []# [[]] * batch_size
        for b in range(batch_size):
            new_beam = []
            start, end = b * beam_size, (b + 1) * beam_size
            scores = probs[start:end]            # (K, V)
            # 加上历史得分
            hist_score = torch.tensor([beams[b][k].score for k in range(beam_size)],
                                      device=device)  # (K,)

            topk_p, topk_t = scores.topk(beam_size, dim=-1)#topk_p, topk_flat = torch.topk(flat, beam_size) 
            total = hist_score.unsqueeze(1) + topk_p   # (K, V)
            score_flat = total.view(-1)
            
            topk_s_flat, topk_t_flat = score_flat.topk(beam_size) if not stochastic else gumbel_topk(score_flat, beam_size)
            beam_idx = topk_t_flat // beam_size  # // vocab_size           # 来自哪条旧 beam
            tok_idx  = topk_t_flat % beam_size  # % vocab_size           # 新 token
            for past_per_layer in past:
                past_per_layer[0][start:end,:,:,:], past_per_layer[1][start:end,:,:,:] = past_per_layer[0][start:end,:,:,:].index_select(0,beam_idx), past_per_layer[1][start:end,:,:,:].index_select(0, beam_idx)
            # if b==1:
            #     torch.save({
            #         'scores':scores,
            #         'hist_scores':hist_score,
            #         'beam':[i.tokens for i in beams[b]]
                    
            #     },f'step{step}.pt')

            for rank in range(beam_size): 
                old_i = beam_idx[rank].item()
                new_i = tok_idx[rank].item()
                new_tok = topk_t[old_i][new_i].item()
                new_score = topk_s_flat[rank].item()
                old_hyp = beams[b][old_i]
                # 已结束的 beam 直接复制，不再扩展
                if old_hyp.tokens[-1] == eos:
                    # next_beams[b].append(old_hyp)
                    new_beam.append(old_hyp)
                else:
                    # next_beams[b].
                    new_beam.append(
                        BeamHypothesis(old_hyp.tokens+[new_tok],
                                       new_score,)) # 这里可再细化存 KV
            # 按得分排序
            new_beam.sort(key=lambda x: x.score / (len(x) ** length_penalty), reverse=True)
            # live_scores = [hyp.score / (len(hyp.tokens)**length_penalty) for hyp in new_beam if hyp.tokens[-1] != eos]
            # done_scores = [hyp.score / (len(hyp.tokens)**length_penalty) for hyp in new_beam if hyp.tokens[-1] == eos]
            next_beams.append(new_beam)
            # if b==batch_size-1 and len(next_beams)==1:
            #     pass
            # if live_scores and done_scores and max(live_scores) < max(done_scores):
            #     beams = next_beams
                # break  
        # if len(next_beams)==1:
        #     print(step, b)
        beams = next_beams




    return [[torch.tensor(hyp.tokens) for hyp in b] for b in beams]


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
checkpoint = 'checkpoints/ir2mol/Transformer/2025-09-23_20_45/470_acc_4604.pth'#'checkpoints/raman2mol/Transformer/2025-09-16_20_19/1110_acc_1705.pth'
ds = checkpoint.split('/')[1]
device = torch.device('cuda:3')
model = load_net_state(model, torch.load(checkpoint, map_location=device, weights_only=True)['model_state']).to(device)
# data = pd.read_pickle('datasets/raman2mol/test.pkl')
# x, y = data['spectrum'], data['smiles']
# dataset = MyDataset(x, y)
loader = make_testloader(ds,2)
lens = []
temps = [1, 1.5, 2]
beamsizes = [3,5,7,9]
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
            # result.append(eval_canonical_smiles([get_smiles(o) for o in pred], 
            #                                     [get_smiles(t) for t in data_iter.true_seq]))
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
                # print(f'{p+1}: Prediction: {get_smiles(pred[p])}, Target: {get_smiles(target)}')
        recalls.append(sum(top_k_result)/sum(lens))
        print(f'beams={beam_size}, temp={temperature}, topk recall={sum(top_k_result)/sum(lens)}')
torch.save(recalls, 'beamsearch.pt')


# greedy_decode
# num_correct =  0
# truelist, falselist = [], []
# for pred_batch, target_batch in zip(result, loader):
#     for pred, target in zip(pred_batch, DataGenerator(target_batch, src_length, device).tgt_y):
#             p, t = get_smiles(pred), get_smiles(target)
#             print(f'Prediction: {p}, Target: {t}')
#             num_correct += p==t
#             if p==t: truelist.append([p,t])
#             else: falselist.append([p,t])
# print(num_correct)
# pd.to_pickle({'true':truelist, 'false':falselist},'search.pkl')
# for i in result:
#     for smiles in i:print(smiles)

