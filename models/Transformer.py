import sys
sys.path.append('/data/YantiLiu/projects/substructure-ID/datasets')

from Transformer_modules import LayerNorm, EncoderLayer, DecoderLayer, MultiHeadedAttention, PositionwiseFeedForward, LearnablePositionalEncoding, NoamOpt, run_epoch, Variable, SimpleLossCompute, greedy_decode, LabelSmoothing, Collator
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
        x = self.encoding(x).transpose(1, 2)  # B, C, L -> B, L, C
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
    
def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    import copy
    from modules import PositionalEncoding, EncoderDecoder, Encoder, Decoder, Embeddings, Generator
    "Helper: Construct a model from hyperparameters."
    spec_encoding = SpectralEncoding(d_model, 8, LayerNorm)
    src_length = spec_encoding(torch.randn([1,1,src_vocab])).shape[-2]
    spec_pos_encoding = LearnablePositionalEncoding(
        d_model, dropout)
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                     c(ff), dropout), N),
        src_embed=nn.Sequential(spec_encoding, spec_pos_encoding),
        tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        generator=Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model, src_length
   
if __name__ == "__main__":
    # from thop import profile
    params = {'conv_ksize':3, 
            'conv_padding':1, 
            'conv_init_dim':32, 
            'conv_final_dim':256, 
            'conv_num_layers':4, 
            'mp_ksize':2, 
            'mp_stride':2, 
            'fc_dim':1024, 
            'fc_num_layers':0, 
            'mixer_num_layers':4,
            'n_classes':957,
            'use_mixer':True,
            }
    x = torch.rand(3, 1, 1024)
    a = torch.tensor([1, 2, 3,4,5])        
    b = torch.tensor([4, 5,6,8])           
    c = torch.tensor([9,10])              
    y = [a,b,c]
    model, src_length = make_model(1024, 11)
    data_generator = [Collator(x, y, src_length)]
    # y = model(src, tgt, src_mask, tgt_mask)
    criterion = LabelSmoothing(size=11, padding_idx=0, smoothing=0.0)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(data_generator, model,
                    SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_generator, model,
                        SimpleLossCompute(model.generator, criterion, None)))

    model.eval()
    src = torch.randn(1,1,1024)
    src_mask = Variable(torch.ones(1, 1, src_length))
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
