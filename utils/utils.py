import logging, random, torch, os
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from utils.dataloader import make_trainloader, make_testloader
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.autograd import Variable
from models.Transformer_modules import run_epoch, DataGenerator
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

import torch.nn.functional as F
from functools import partial
import json


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


def write(x, path):
    
    with open(path, 'a') as file:
        for i, l in zip(x[0],x[1]):
            file.write(f'{i}:{l}\n')
        
        
def eval_canonical_smiles(pred, target):
    
    true = 0
    false = 0
    invalid = 0
    for i, j in zip(pred, target):
        try:
            canonical_pred = Chem.MolToSmiles(Chem.MolFromSmiles(i))
            if canonical_pred == j: 
                true += 1
            else:
                false += 1
        except:
            invalid += 1
            continue
    return true


def top_k_eval(pred, target, k):
    if not len(pred)==k:print(len(pred),k)
    assert len(pred) == k
    true_dict = np.zeros(k)
    true_dict+=np.array([get_smiles(p)==get_smiles(target) for p in pred])
    # for i in range(1, k):
    #     true_dict[i] += true_dict[i-1]
    return true_dict


def eval_tokens(prediction, target):
    true = 0
    for p, t in zip(prediction, target):
        for i in range(len(t)):
            try:
                if p[i]==t[i]:
                    true += 1
            except IndexError: break
    return true


def ddp_setup(rank: int, world_size: int, gpu_ids: str):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
        gpu_ids: GPU indices like "2,3"
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    torch.cuda.set_device(rank)
    os.environ["MASTER_PORT"] = '29511'
    os.environ["MASTER_ADDR"] = "localhost"
    try:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    except RuntimeError:
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size,
                                init_method="env://?use_libuv=False")
   
   
def cleanup():
    dist.barrier()
    dist.destroy_process_group()
    
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)# if you are using multi-GPu.

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

def get_spec_mask(tensor, min_ratio=0.15, max_ratio=0.30, device='cpu'):
    assert tensor.dim() >= 2,'Input must be n less than 2D'
    m, n = tensor.shape

    ratios = torch.empty(m, device=tensor.device).uniform_(min_ratio, max_ratio)
    num_zeros = torch.clamp((ratios * n).round().long(), max=n - 1)

    zero_mask = torch.zeros(m, n, dtype=torch.bool, device=tensor.device)
    for i in range(m):
        if num_zeros[i] > 0:
            idx = torch.randperm(n, device=tensor.device)[:num_zeros[i]]
            zero_mask[i, idx] = True

    tensor.masked_fill_(zero_mask, 0)

    return tensor.to(device), zero_mask.to(device)

    
class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStop:
    def __init__(self, patience=5, mode='max', delta=0.0001, rank=0):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.rank = rank
        self.epoch = 0
        if self.mode == 'min':
            self.val_score = np.inf
        else:
            self.val_score = -np.inf

    def __call__(self, epoch_score, model_path, model, optimizer, lr_scheduler):
        if self.rank == 0:
            if self.mode == 'min':
                score = -1. * epoch_score
            else:
                score = np.copy(epoch_score)
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(epoch_score, model_path, model, optimizer, lr_scheduler)
            elif score < self.best_score + self.delta:
                self.counter += 1
                # print(f'Early stopper count: {self.counter}/{self.patience}')
                # if self.counter >= self.patience:
                #     self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(epoch_score, model_path, model, optimizer, lr_scheduler)
                self.counter = 0
            self.early_stop = torch.tensor([int(self.counter >= self.patience)]).cuda()
        else:
            self.early_stop = torch.tensor([0]).cuda()
        self.epoch += 1
        dist.broadcast(self.early_stop, src=0)
        # return epoch_score
    
    def save_checkpoint(self, epoch_score, model_path, model, optimizer, lr_scheduler):
            # epoch = int(model_path.split('/')[-1].split('_')[0])
            # # if self.epoch <= 1000:
            # if epoch - self.epoch >= 50:
            #     flag = True
            #     self.epoch = epoch
            # # else: flag = True
            # if flag:
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'lr_scheduler_state': lr_scheduler.state_dict(),
            'epoch': self.epoch,
            }, model_path)
        try: model.argue()
        except: pass
        self.val_score = epoch_score
            # else: pass


class Engine:
    def __init__(self, train_loader=None, val_loader=None, test_loader=None,
                 criterion=None, optimizer=None, lr_scheduler=None, device='cpu',
                 model=None, eval_sample_num=None, **kwargs):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.src_length = kwargs['src_length']
        mode = kwargs['mode']
        self.pretrain = 1 if 'pretrain' in mode else 0
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_sample_num = eval_sample_num
        self.kwargs = kwargs
        # if nn.BatchNorm1d in list(model.modules()):
        #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
        if not mode == 'test': 
            rank = kwargs['rank'] 
            self.model = DDP(model.to(rank), device_ids=[rank], find_unused_parameters=False)
            self.device = self.model.device
            self.rank = rank
        else: 
            self.device = device
            self.model=model.to(device)
        
    def train_epoch(self, epoch, binary=False):
        losses = AverageMeter()
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)
        bar = tqdm(self.train_loader, ncols=100) if self.rank == 0 else self.train_loader
        
        for _, dataset in enumerate(bar):
            self.optimizer.zero_grad()
            if self.pretrain:
                data, mask = get_spec_mask(dataset[0].squeeze(1), device=self.device)
                target = torch.masked_fill(data, mask==0, 0)
                output = self.model(data.unsqueeze(1))
                loss = self.criterion(output, target.to(self.model.device))
                batch_size = target.shape[0]
            else:  
                data_generator = DataGenerator(dataset, self.src_length, self.device)
                output, loss = run_epoch(data_generator, self.model, self.criterion)
                # print(torch.max(output[0],dim=1)[1])
                batch_size = data_generator.batch_size
                self.train_gen = data_generator
            if binary:
                output = torch.sigmoid(output)
            loss.backward()
            self.optimizer.step()
            losses.update(loss.data.item(), batch_size)
            if self.rank == 0:
                bar.set_description(
                    f'Epoch{epoch:3d}, train loss:{losses.avg:6f}')
        logging.info(f'Epoch{epoch:3d}, train loss:{losses.avg:6f}')
        return losses.avg

    def evaluate_epoch(self, epoch, binary=False):
        token_accs = AverageMeter()
        smiles_accs = AverageMeter()
        losses = AverageMeter()
        flag = True
        self.model.eval()
        bar = tqdm(self.val_loader, ncols=125) if self.rank == 0 else self.val_loader
        with torch.no_grad():
            for _, dataset in enumerate(bar):
                if self.pretrain:
                    data, mask = get_spec_mask(dataset[0].squeeze(1), device=self.device)
                    target = torch.masked_fill(data, mask==0, 0)
                    output = self.model(data.unsqueeze(1), None).masked_fill_(mask==0, 0)
                    loss = self.criterion(output, target.to(self.model.device))
                    batch_size = output.shape[0]
                else: 
                    data_generator = DataGenerator(dataset, self.src_length, self.device)
                    output, loss = run_epoch(data_generator, self.model, self.criterion)
                    batch_size = data_generator.batch_size
                if binary:
                    output = torch.sigmoid(output)
                # loss = self.criterion(output, target.to(self.model.device))
                losses.update(loss.item(), batch_size)
                output = output.detach().cpu().numpy()
                
                prediction = []
                if self.pretrain:
                    if self.rank==0:
                        bar.set_description(f'Epoch{epoch:3d}, eval loss:{losses.avg:6f}')
                else:
                    flag = True
                    if epoch % 10 == 0:
                        # data_generator = self.train_gen
                        output = self.greedy_decode(data_generator, **self.kwargs['greedy_decode'])
                        for out in output:
                            prediction+=out
                        target = data_generator.true_seq#[:self.eval_sample_num]
                        token_acc = eval_tokens(output, target) / np.sum([len(i) for i in target])
                        smiles_acc = eval_canonical_smiles([get_smiles(o) for o in output], [get_smiles(t) for t in target]) / len(target)
                        token_accs.update(token_acc, len(target)) # accuracy.item()
                        smiles_accs.update(smiles_acc, len(target))
                        if self.rank==0:
                            bar.set_description(f'Epoch{epoch:3d}, eval loss:{losses.avg:6f} , token accuracy:{token_accs.avg:6f}, smiles accuracy:{smiles_accs.avg:6f}')  
                            logging.info(f'Epoch{epoch:3d}, eval loss:{losses.avg:6f} , token accuracy:{token_accs.avg:6f}, SMILES accuracy:{smiles_accs.avg:6f}')
                    else:
                        if self.rank==0:
                            bar.set_description(
                                f'Epoch{epoch:3d}, eval loss:{losses.avg:6f}')
        self.lr_scheduler.step()
        # if target.ndim == 1:
        #     logging.info(f'Epoch{epoch:3d}, eval loss:{losses.avg:6f}, eval accuracy:{token_accs.avg:6f}')
        # else:
        #     logging.info(f'Epoch{epoch:3d}, eval loss:{losses.avg:6f}')
        return losses.avg, smiles_accs.avg, prediction if flag else None

    def test_epoch(self, binary=False):
        token_accs = AverageMeter()
        losses = AverageMeter()
        self.model.eval()
        bar = tqdm(self.test_loader, ncols=100)
        outputs = []
        predicted = []
        true = []
        accuracy = 0
        with torch.no_grad():
            for _, dataset in enumerate(bar):
                data_generator = DataGenerator(dataset, self.src_length, self.device)
                if binary:
                    data, target = dataset
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    target = target.detach().cpu().numpy()
                    output = torch.sigmoid(output)
                    prediction = torch.greater_equal(output, 0.5).to(torch.float64).cpu().detach().numpy()
                    accuracy = metrics.f1_score(target, prediction, 
                                                average='macro', zero_division=0.0)
                    losses.update(loss.item(), data.size(0))
                else:
                    prediction = self.greedy_decode(data_generator, **self.kwargs['greedy_decode'])
                    target = [get_smiles(t) for t in data_generator.tgt_y]
                    result = eval_canonical_smiles(prediction, target)
                    accuracy = result / len(prediction)
                token_accs.update(accuracy, len(prediction))
                bar.set_description(
                    f"Accuracy:{token_accs.avg:.5f}")
                predicted += prediction.tolist() if type(prediction) != list else prediction
                true += target.tolist() if type(target) != list else target
        return outputs, predicted, true, token_accs.avg, losses.avg


    def greedy_decode(self, data_generator, bos=0, eos=2, max_len=30, repetition_penalty=None):
        print(repetition_penalty)
        if type(self.model) == torch.nn.parallel.DistributedDataParallel:
            model = self.model.module
        else:
            model = self.model 
        predlist = []
        # spec = torch.FloatTensor(spec[::-1].copy())
        # spec = spec.reshape(1, 1, spec.shape[-1]) if spec.dim() != 3 else spec
        for i in range(len(data_generator.src)):
            src, src_mask = data_generator.src[i].unsqueeze(0), Variable(torch.ones(1, 1, data_generator.src_length, device=data_generator.device))
            ys = torch.ones(1, 1, device=data_generator.device, dtype=torch.long).fill_(bos)
            memory = model.encode(src, src_mask)
            for step in range(max_len):
                tgt_mask = torch.ones_like(ys, device=data_generator.device)
                out = model.decode(memory, src_mask,
                                Variable(ys),
                                Variable(tgt_mask))
                out = out[0] if type(out) == tuple else out                        
                logits = model.generator(out[:, -1])
                
                if repetition_penalty is not None:
                    tok_seen = list(set(ys[0].tolist()))
                    for tok in tok_seen:
                        logits[:, tok] /= repetition_penalty
                        
                prob = F.softmax(logits, dim=-1)
                _, next_word = torch.max(prob, dim=-1)
                next_word = next_word.data[0]
                if step != max_len-1:
                    ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=data_generator.device).fill_(next_word)], dim=1)                
                    if next_word != eos: pass
                    else:                 
                        predlist.append(ys[0][1:])
                        break
                else:
                    ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=data_generator.device).fill_(eos)], dim=1)
                    predlist.append(ys[0][1:])
                    break
        return [get_smiles(y) for y in predlist]


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


def load_optm_state(optimizer, state_dict):
    for state in optimizer.state.values():
        for k,v in state_dict.items():
            if isinstance(v, torch.Tensor):state[k]= v.cuda()
            

def get_global_score(score):
    score = torch.tensor(score).cuda()
    dist.reduce(score, dst=0, op=dist.ReduceOp.SUM)  # 例：求平均/求和
    global_score = score.item() / dist.get_world_size()
    return global_score


def train_model(model, save_path, ds, **kwargs):
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.tensorboard import SummaryWriter
    
    if kwargs['rank'] == 0:
        writer = SummaryWriter(log_dir = save_path)
    ddp_setup(kwargs['rank'], kwargs['world_size'], kwargs['gpu_ids'])
    train_loader, val_loader = make_trainloader(ds, batch_size=kwargs['batch_size'],
                                                train_size=kwargs['train_size'],seed=42, mode=kwargs['mode'], )

    # test_loader = make_testloader(ds, ) if not 'pretrain' in kwargs['mode'] else None
    binary = False  #True if 'ir' in ds or 'raman' in ds else 
    if 'pretrain' in kwargs['mode']:
        criterion = nn.MSELoss()
    elif binary:
        criterion = torch.nn.BCELoss()
    else:
        # criterion = LabelSmoothing(padding_idx=0, 
        #                            smoothing=0.0, 
        criterion = partial(F.cross_entropy, ignore_index=1, reduction='none')
        
    optimizer = torch.optim.AdamW(
        model.parameters(), **kwargs['Adam_params'])
    if kwargs['checkpoint']:
        load_optm_state(optimizer, kwargs['checkpoint']['optimizer_state'])
        
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=6e-5)
    if kwargs['checkpoint']:
        lr_scheduler.load_state_dict(kwargs['checkpoint']['lr_scheduler_state'])
        
    mode = 'max' if not 'pretrain' in kwargs['mode'] else 'min'
    es = EarlyStop(patience=kwargs['patience'], mode=mode, rank=kwargs['rank'],)
    engine = Engine(train_loader=train_loader, val_loader=val_loader, 
                    criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, model=model, **kwargs)
    # start to train 
    for epoch in range(kwargs['epoch']):
        dist.barrier()
        train_loss = engine.train_epoch(epoch, binary=binary)
        val_loss, acc, seq_pred = engine.evaluate_epoch(epoch, binary=binary)
        val_loss_global, val_acc_global, train_loss_global = get_global_score(val_loss), get_global_score(acc), get_global_score(train_loss)
        if 'pretrain' in kwargs['mode']:
            es(val_loss_global, f'{save_path}/{epoch}_vloss_{str(val_loss)[2:6]}.pth', model, optimizer, lr_scheduler)
        elif acc is not None:
            es(acc, f'{save_path}/{epoch}_acc_{str(acc)[2:6]}.pth', model, optimizer, lr_scheduler)
        else:
            assert ValueError('not metrics')

        if kwargs['rank'] == 0:
            writer.add_scalar("Loss/train", train_loss_global, epoch)
            writer.add_scalar("Loss/val", val_loss_global, epoch)
            if val_acc_global:
                writer.add_scalar("Accuracy/val", val_acc_global, epoch)
            if seq_pred:
                writer.add_histogram('Token Frequency/val', torch.stack(seq_pred), epoch, bins=1)
        if es.early_stop == 1:
            break
        dist.barrier()
    cleanup()


def test_model(model, ds, device='cpu', verbose=True, **kwargs):
    
    test_loader = make_testloader(ds)
    binary = False#True if 'ir' in ds or 'raman' in ds else 
    criterion = torch.nn.BCELoss() if binary else partial(F.cross_entropy, ignore_index=1, reduction='none')
    engine = Engine(test_loader=test_loader,
                    criterion=criterion, model=model, device=device, **kwargs)
    outputs, pred, true, _, _ = engine.test_epoch(binary=binary)

    if verbose:
        # print(metrics.classification_report(true, pred, digits=4))
        smiles_result = eval_canonical_smiles(pred, true)
        print('Recall: %.4f\n' %(smiles_result/len(pred)))
        logging.info('Recall: %.4f\n' %(smiles_result/len(pred)))
        # print('False SMILES: %.4f\n' %(smiles_result[1].item()/len(pred)))
        # print('Invalid SMILES: %.4f\n' %(smiles_result[2].item()/len(pred)))
        # print('Token accuracy: %.4f\n' %(eval_tokens(pred, true)/len(pred)))
        # logging.info(metrics.classification_report(true, pred, digits=4))
        # logging.info(f'accuracy:{metrics.accuracy_score(true, pred):.5f}')
        # logging.info('Exact match rate (EMR): %.4f\n' %metrics.accuracy_score(true, pred))
        # logging.info('Precision: %.4f\n' %(np.count_nonzero(true==pred)/pred.shape[0]/pred.shape[1]))
    return outputs, pred, true

def inf_time(model):
    iterations = 300
    device = torch.device("cuda:0")
    model.to(device)

    random_input = torch.randn(1, 1, 1024).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # Preheat GPU
    for _ in range(50):
        _ = model(random_input)

    # Measure inference time
    times = torch.zeros(iterations)     # Save the time of each iteration
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = model(random_input)
            ender.record()
            # Synchronize GPU time
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # Calculate time
            times[iter] = curr_time
            # print(curr_time)

    mean_time = times.mean().item()
    return mean_time
    # print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))
