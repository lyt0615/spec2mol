import sys
sys.path.append('/data/YantiLiu/projects/substructure-ID/datasets')

try: 
    from Transformer_modules import LayerNorm, EncoderLayer, DecoderLayer, MultiHeadedAttention, PositionwiseFeedForward, LearnablePositionalEncoding, NoamOpt, run_epoch, Variable, LabelSmoothing, DataGenerator, LearnableClassEmbedding, PositionalEncoding, EncoderDecoder, Encoder, Decoder, Embeddings, Generator
except ModuleNotFoundError:
    from models.Transformer_modules import LayerNorm, EncoderLayer, DecoderLayer, MultiHeadedAttention, PositionwiseFeedForward, LearnablePositionalEncoding, NoamOpt, run_epoch, Variable, LabelSmoothing, DataGenerator, LearnableClassEmbedding, PositionalEncoding, EncoderDecoder, Encoder, Decoder, Embeddings, Generator
import torch
import torch.nn as nn
import numpy as np


class SpectralEncoding(nn.Module):
    def __init__(self, d_model, patch_size, norm_layer):
        super().__init__()
        self.d_model = d_model
        self.encoding = nn.Conv1d(
            1, d_model, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = norm_layer(d_model) if norm_layer else nn.Identity()

    def forward(self, x):
        # try:
        x = self.encoding(x).transpose(1, 2)  # B, C, L -> B, L, C
        # except RuntimeError:print(x.shape)
        return self.norm(x)


class FPGrowingModule(nn.Module):
    """FPGrowingModule.

    Accept an input hidden dim and progressively grow by powers of 2 s.t.

    We eventually get to the final output size...

    """

    def __init__(
        self,
        hidden_input_dim: int = 256,
        final_target_dim: int = 4096,
        num_splits=4,
        reduce_factor=2,
    ):
        super().__init__()

        self.hidden_input_dim = hidden_input_dim
        self.final_target_dim = final_target_dim
        self.num_splits = num_splits
        self.reduce_factor = reduce_factor

        final_output_size = self.final_target_dim

        # Creates an array where we end with final_size and have num_splits + 1
        # different entries in it (e.g., num_splits = 1 with final dim 4096 has
        # [2048, 4096])
        layer_dims = [
            int(np.ceil(final_output_size / (reduce_factor**num_split)))
            for num_split in range(num_splits + 1)
        ][::-1]

        # Start by predicting into the very first layer dim (e.g., 256  -> 256)
        self.output_dims = layer_dims

        # Define initial predict module
        self.initial_predict = nn.Sequential(
            nn.Linear(
                hidden_input_dim,
                layer_dims[0],
            ),
            nn.Sigmoid(),
        )
        predict_bricks = []
        gate_bricks = []
        for layer_dim_ind, layer_dim in enumerate(layer_dims[:-1]):
            out_dim = layer_dims[layer_dim_ind + 1]

            # Need to update nn.Linear layer to be fixed if the right param is
            # called
            lin_predict = nn.Linear(layer_dim, out_dim)
            predict_brick = nn.Sequential(lin_predict, nn.Sigmoid())

            gate_bricks.append(
                nn.Sequential(
                    nn.Linear(hidden_input_dim, out_dim), nn.Sigmoid())
            )
            predict_bricks.append(predict_brick)

        self.predict_bricks = nn.ModuleList(predict_bricks)
        self.gate_bricks = nn.ModuleList(gate_bricks)

    def forward(self, hidden):

        cur_pred = self.initial_predict(hidden)
        output_preds = [cur_pred]
        for _out_dim, predict_brick, gate_brick in zip(
            self.output_dims[1:], self.predict_bricks, self.gate_bricks
        ):
            gate_outs = gate_brick(hidden)
            pred_out = predict_brick(cur_pred)
            cur_pred = gate_outs * pred_out
            output_preds.append(cur_pred)
        return output_preds
    
    
def make_model(tgt_vocab, src_vocab=1024, N=6, d_model=512, d_ff=1024, h=8, dropout=0.1, mode='train'):
    import copy
    "Helper: Construct a model from hyperparameters."
    spec_encoding = SpectralEncoding(d_model, 8, LayerNorm)
    mol_embedding = Embeddings(d_model)
    position = PositionalEncoding(d_model, dropout)
    cls_embedding = LearnableClassEmbedding(d_model, dropout)
    # spec_pos_embedding = PositionalEncoding(d_model, dropout)
    # spec_cls_embedding = LearnableClassEmbedding(d_model, dropout)
    src_length = position(cls_embedding(spec_encoding(torch.randn([1,1,src_vocab])))).shape[-2]
    c = copy.deepcopy
    # mol_cls_embedding = c(spec_cls_embedding)
    
    src_embed=nn.Sequential(c(spec_encoding), c(cls_embedding), c(position))
    tgt_emb = nn.ModuleList([c(mol_embedding), c(position)])
    
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                     c(ff), dropout), N),
        src_embed=src_embed,
        tgt_emb=tgt_emb,
        generator=Generator(d_model, tgt_vocab),
        mode=mode)
    # if mode =='pretrain_spec':
    #     model = SpecEncoder(encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
    #                 generator=Generator(d_model, tgt_vocab),
    #                 src_embed=nn.Sequential(spec_encoding, spec_pos_encoding),)
    # if mode == 'pretrain_mol':
    #     model = nn.Sequential([layer[0] for layer in model.decoder.sublayer].next(), model.generator)   
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model, src_length
   
if __name__ == "__main__":
    # from thop import profile
    x = torch.rand(3, 1, 1024)
    a = torch.tensor([1, 8, 3,4,5, 2])        
    b = torch.tensor([1,4, 6,8,2,0])           
    c = torch.tensor([1,10,2,0,0,0])              
    y = torch.vstack([a,b,c])
    model, src_length = make_model(11, 1024, mode='pretrain_spec')
    data_generator = DataGenerator([x, y], src_length, torch.device('cpu'))
    src, tgt, src_mask, tgt_mask = data_generator.src, data_generator.tgt, None, data_generator.tgt_mask
    # y = model(src, tgt, src_mask, tgt_mask)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=1)
    # criterion = LabelSmoothing(size=11, padding_idx=0, smoothing=0.0)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(data_generator, model,
                    criterion)
        model.eval()
        print(run_epoch(data_generator, model,
                       criterion))

    model.eval()
    src = torch.randn(1,1,1024)
    src_mask = Variable(torch.ones(1, 1, src_length))
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=0))
