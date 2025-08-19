import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.autograd import Variable


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
    
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, mode='train'):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.mode = mode
        if mode == 'pretrain_spec':
            for p in self.decoder.parameters():
                p.requires_grad = False
            for p in self.tgt_embed.parameters():
                p.requires_grad = False
                
        if mode == 'pretrain_mol':
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.src_embed.parameters():
                p.requires_grad = False
            for layer in self.decoder.layers:
                for sublayer in layer.sublayer[1:]:
                    for p in sublayer.parameters():
                        p.requires_grad = False
                          
    def forward(self, src=None, tgt=None, src_mask=None, tgt_mask=None):
        if self.mode == 'train' or self.mode == 'test':
            "Take in and process masked src and target sequences."
            return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
        elif self.mode == 'pretrain_spec':
            return self.generator(self.encode(src, None)[:, 0])
        elif self.mode == 'pretrain_mol':
            tgt = self.tgt_embed(tgt)
            for layer in self.decoder.layers:
                x = layer.sublayer[0](tgt, lambda x: layer.self_attn(x, x, x, None))
            return self.generator(x[:, 0])
        
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class SpecEncoder(nn.Module):

    def __init__(self, encoder, src_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator
    def forward(self, src, src_mask):
        return self.generator(self.encoder(self.src_embed(src), src_mask)[:, 0])
    
    
class MolEncoder(nn.Module):
    """
    只保留 DecoderLayer 的第一层：Masked Self-Attention
    输入：  tgt      [batch, tgt_len, d_model]
           tgt_mask [batch, 1, tgt_len, tgt_len]  # 下三角 0/1 掩码
    输出：  out      [batch, tgt_len, d_model]
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        # 复制原结构的 Norm + Dropout + Residual
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 重新创建一个 MultiHeadAttention（与原 DecoderLayer 的 self_attn 同配置）
        self.self_attn = MultiHeadedAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True   # PyTorch ≥1.9
        )

    def forward(self, x, tgt_mask=None):
        # Norm
        normed = self.norm(x)
        # 带掩码的自注意力
        attn_out, _ = self.self_attn(normed, normed, normed,
                                     attn_mask=tgt_mask)   # tgt_mask 就是下三角
        # Residual + Dropout
        out = x + self.dropout(attn_out)
        return out
    
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x) # F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)



class LearnablePositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the learnable positional embeddings
        self.pe = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class LearnableClassEmbedding(nn.Module):
    "Implement the class and distillation embeddings."

    def __init__(self, d_model, dropout, dist=False):
        super(LearnableClassEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the learnable embeddings
        self.ce = nn.Parameter(torch.randn(1, d_model))
        self.de = nn.Parameter(torch.randn(1, d_model)) if dist else None

    def forward(self, x):
        # concatenate the class and distillation embeddings
        if self.de is not None:
            x = torch.cat((self.ce.repeat(x.size(0), 1).unsqueeze(1), x,
                        self.de.repeat(x.size(0), 1).unsqueeze(1)), dim=1)
        else:
            x = torch.cat((self.ce.repeat(x.size(0), 1).unsqueeze(1), x), dim=1)
        return x
    
def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1, mode='train'):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                     c(ff), dropout), N),
        src_embed=nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        generator=Generator(d_model, tgt_vocab),
        mode=mode)
    return model
        
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Collator:
    def __init__(self, batch, src_length, device=torch.device('cpu')):
        self.device = device
        self.src, self.tgt, self.src_mask, self.tgt_mask, self.tgt_y, self.ntokens = [i.to(self.device) for i in self.collate(batch, src_length)]
        self.batch_size = self.src.size(0)
        self.true_seq = [seq[torch.where(seq)[0][1:]] for seq in batch[1]]
        
    def make_pad_mask(self, lengths: torch.Tensor, max_len: int = None):
        """
        lengths: [batch_size], 每个样本的实际长度
        return: [batch_size, 1, 1, max_len] 的 bool mask，True 表示 PAD 位置
        """
        # batch_size = lengths.size(0)
        max_len = lengths.max()
        # [batch, max_len]
        mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
        return mask.unsqueeze(1).to(self.device)

    def make_causal_mask(self, sz: int):
        return torch.triu(torch.ones(sz, sz, dtype=torch.int, device=self.device), diagonal=1)
    def collate(self, batch, src_length=None, padding_value=0):
        if src_length is None:
            src_lengths = torch.tensor([k.shape[-1] for k in batch[0]])
        else:
            src_lengths = torch.tensor([src_length for _ in batch[0]])
        #     src_mask=None
        #     src = batch[0]
        # else:
        src = torch.nn.utils.rnn.pad_sequence(batch[0], batch_first=True, padding_value=padding_value)
        src_mask = self.make_pad_mask(src_lengths, src.size(-1))   # [B,1,1,Ls]
        tgt = batch[1] # [:,:-1]
        tgt_y = batch[1][:,1:]
        ntokens = sum([(y!=padding_value).sum() for y in tgt_y])
        # tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=padding_value)
        # tgt_y = torch.nn.utils.rnn.pad_sequence(tgt_y, batch_first=True, padding_value=padding_value)
        tgt_lengths = torch.tensor([j.shape[-1] for j in tgt])
        tgt_pad_mask = self.make_pad_mask(tgt_lengths, tgt.size(-1))  # [B,1,1,Lt]
        causal_mask = self.make_causal_mask(tgt.size(1))  # [Lt,Lt]
        tgt_mask = tgt_pad_mask | causal_mask #tgt_pad_mask  .unsqueeze(0).unsqueeze(0)
        return src, tgt, src_mask, tgt_mask, tgt_y, ntokens
    
class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    # start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    # t = (data_iter.src, data_iter.tgt,
                        # data_iter.src_mask, data_iter.tgt_mask)
    # for i in range(3):
    #     try:print(t[i].shape)
    #     except AttributeError:print('***',i)
    # for i, batch in enumerate(data_iter):
    out = model.forward(data_iter.src, data_iter.tgt,
                        data_iter.src_mask, data_iter.tgt_mask)
    out = F.softmax(out, dim=-1)
    out = model.module.generator(out)
    # shift_logits = out[:, :-1, :].contiguous().view(-1, 24)
    # shift_labels = data_iter.tgt[:, 1:].contiguous().view(-1)
    # print('shift_labels[:10] =', shift_labels[:10].tolist())
    # print('shift_logits[0, eos] =', shift_logits[0, 23].item())
    # print('generator.bias[eos] =', model.module.generator.proj.bias[23].item())
    # print('generator.bias.mean() =', model.module.generator.proj.bias.mean().item())    
    loss = loss_compute(out, data_iter.tgt_y, data_iter.ntokens)
    total_loss += loss
    total_tokens += data_iter.ntokens
    # if i % 50 == 1:
    #     elapsed = time.time() - start
    #     print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
    #             (i, loss / data_iter.ntokens, tokens / elapsed))
    #     start = time.time()
    #     tokens = 0
    return out, total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.tgt) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, padding_idx, criterion, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = criterion  #nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None

    def forward(self, x, target):
        # assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist.argmax(dim=-1).long()
        return self.criterion(x, Variable(self.true_dist, requires_grad=False))


def data_gen(src, tgt, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        # data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        # data[:, 0] = 1
        # src = src if src is not None else Variable(data, requires_grad=False)
        # tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


class LossCompute:
    "A simple loss compute and train function."

    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        loss = self.criterion(x[:, :-1, :].contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)).sum() / norm
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss * norm


def greedy_decode(model, data_generator, max_len, start_symbol=0, sample_num=None):
    import pandas as pd
    if type(model) == torch.nn.parallel.DistributedDataParallel:
        model = model.module 
    predlist = []
    eos = data_generator.tgt_y.max()
    # lengths = [len(seq) for seq in data_generator.true_seq]
    # print('mean', np.mean(lengths), 'max', np.max(lengths), 'pct 90', np.percentile(lengths, 90))
    sample_num = sample_num if sample_num else len(data_generator.src)
    for i in range(sample_num):
        src, src_mask = data_generator.src[i].unsqueeze(0), Variable(torch.ones(1, 1, 128)).to(data_generator.device)
        ys = torch.ones(1, 1, device=src.device, dtype=torch.long).fill_(start_symbol)
        memory = model.encode(src, Variable(src_mask))
        for step in range(max_len):
            tgt_mask = Variable(subsequent_mask(step+1).type_as(data_generator.tgt_mask))
            out = model.decode(memory, src_mask,
                            Variable(ys),
                            tgt_mask)
            prob = F.softmax(model.generator(out[:, -1]), dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            next_word = next_word.data[0]
            # ys[:, i+1] = next_word
            ys = torch.cat([ys, torch.ones(1, 1).fill_(next_word).type_as(data_generator.tgt)], dim=1)
    #         print(f"mask shape={tgt_mask.shape} | "
    #   f"logits[:5]={prob[0,:5].tolist()} {prob.topk(5).indices.tolist()}| "
    #   f"chosen={next_word.item()} prob={prob[0,next_word.item()].item():.3f}")
            if next_word == eos:
                break

        predlist.append(ys)
        # print(ys)
    # pd.to_pickle([data_generator.src, data_generator.src_mask, data_generator.tgt,
    #         data_generator.tgt_mask, prob],'data.pkl')
    return predlist


def seed_everything(seed):
    import random
    import os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def generate_mask(query_padding_matrix, key_padding_matrix):
    query_padding_matrix = query_padding_matrix.unsqueeze(-1)
    key_padding_matrix = key_padding_matrix.unsqueeze(1)
    mask = query_padding_matrix | key_padding_matrix
    return mask

if __name__ == "__main__":
    seed_everything(624)
    # Train the simple copy task.
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model,
                  LossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5), model,
                        LossCompute(model.generator, criterion, None)))

    model.eval()
    src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    src_mask = Variable(torch.ones(1, 1, 10))
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
