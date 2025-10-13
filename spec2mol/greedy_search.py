from models.Transformer import make_model
import torch, json
from torch.autograd import Variable
import torch.nn.functional as F
from transformers import AutoTokenizer


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


def collate_fn(batch):
    tokenizer = AutoTokenizer.from_pretrained("models/moltokenizer")
    x = [item[0] for item in batch]
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=1).unsqueeze(1)
    y = [item[1] for item in batch]
    y = tokenizer(y, padding=True, truncation=True, return_tensors="pt")
    return x, y


def load_net_state(net, state_dict):
    '''check the keys and load the weight'''
    net_keys = net.state_dict().keys()
    state_dict_keys = state_dict.keys()
    for key in net_keys:
        if key in state_dict_keys:
            # load the weight
            net.state_dict()[key].copy_(state_dict[key])
        else:
            print('key error: ', key)
    net.load_state_dict(net.state_dict())
    return net


def greedy_decode(model, spec, src_length, max_len=30, bos=0, end_symbol=2):
    if type(model) == torch.nn.parallel.DistributedDataParallel:
        model = model.module 
    predlist = []
    l = []
    spec = torch.FloatTensor(spec[::-1].copy())
    spec = spec.reshape(1, 1, spec.shape[-1]) if spec.dim() != 3 else spec
    for i in range(len(spec)):
        src, src_mask = spec[i].unsqueeze(0), Variable(torch.ones(1, 1, src_length, device=spec.device))
        ys = torch.ones(1, 1, device=src.device, dtype=torch.long).fill_(bos)
        memory = model.encode(src, src_mask)
        for step in range(max_len):
            tgt_mask = torch.ones_like(ys, device=ys.device)
            out = model.decode(memory, src_mask,
                            Variable(ys),
                            Variable(tgt_mask))
            out = out[0] if type(out) == tuple else out
            logits = model.generator(out[:, -1])
            l.append(logits)
            prob = F.softmax(logits, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            next_word = next_word.data[0]
            if step != max_len-1:
                ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long).fill_(next_word)], dim=1)                
                if next_word != end_symbol: pass
                else:                 
                    predlist.append(ys[0][1:])
                    break
            else:
                ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long).fill_(end_symbol)], dim=1)
                predlist.append(ys[0][1:])
                break
    return get_smiles(predlist[0])

if __name__ == "__main__":

    spec = torch.randn(1024) ### input spectrum: 长度1024，波数范围400-4000
    vocab_size = 181
    model, src_length = make_model(vocab_size, N=4, d_model=512)
    beam_size = 10
    checkpoint = 'spec2mol/checkpoints/checkpoint.pth'
    device = torch.device('cpu')
    model = load_net_state(model, torch.load(checkpoint, map_location=device, weights_only=True)['model_state']).to(device)

    pred = greedy_decode(model, 
                spec,
                src_length)
    print(pred)