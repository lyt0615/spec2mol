import logging, random, torch, os
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from utils.dataloader_ddp import make_trainloader, make_testloader
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from models.Transformer_modules import run_epoch,  greedy_decode, DataGenerator
from config import tokens
from rdkit import Chem
from torch.utils.data import RandomSampler


def get_smiles(label):
    smiles = ''
    for l in label:
        smiles += tokens[l]
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
    return np.array([true, false, invalid])


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
    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #     s.bind(('', 0))
    #     master_port = str(s.getsockname()[1])
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
# def get_vib_masked(spec,
#                 mask_ratio=range(15, 30),
#                 # silent_start=1800,   # cm⁻¹
#                 # silent_end=2800,
#                 # wavenumbers=torch.linspace(4000,400,1024),    # 1-D tensor length L
#                 mask_value=0.0
#                 ):
#     B, *_ = spec.shape
#     mask_ratio = RandomSampler(mask_ratio)
#     len_keep = int((1 - mask_ratio) * spec[0].numel())
#     noise = torch.rand(B, spec[0].numel(), device=spec.device)
#     ids_shuffle = torch.argsort(noise, dim=1)
#     ids_restore = torch.argsort(ids_shuffle, dim=1)

#     mask = torch.ones_like(spec.view(B, -1))
#     mask[:, :len_keep] = 0
#     mask = torch.gather(mask, 1, ids_restore).view_as(spec).bool()

#     masked_spec = spec.masked_fill(mask, mask_value)
#         # mask = mask.view_as(spec)
#     return mask, masked_spec

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
                print(f'Early stopper count: {self.counter}/{self.patience}')
                # if self.counter >= self.patience:
                #     self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(epoch_score, model_path, model, optimizer, lr_scheduler)
                self.counter = 0
            self.early_stop = torch.tensor([int(self.counter >= self.patience)]).cuda()
        else:
            self.early_stop = torch.tensor([0]).cuda()
        dist.broadcast(self.early_stop, src=0)
        return epoch_score
    
    def save_checkpoint(self, epoch_score, model_path, model, optimizer, lr_scheduler):
        flag = False
        if self.rank == 0:
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
            self.val_score = epoch_score
            print('🤖: I May... be Paranoid, but... not an... Agent...')
            # else: pass
        else: pass


class Engine:
    def __init__(self, train_loader=None, val_loader=None, test_loader=None,
                 criterion=None, optimizer=None, lr_scheduler=None, device='cpu',
                 model=None, **kwargs):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.src_length = kwargs['src_length']
        self.max_len = kwargs['max_len']
        mode = kwargs['mode']
        self.pretrain = 1 if 'pretrain' in mode else 0
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
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
        accuracies = AverageMeter()
        losses = AverageMeter()
        flag = True
        self.model.eval()
        bar = tqdm(self.val_loader, ncols=100) if self.rank == 0 else self.val_loader
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
                    target = data_generator.tgt_y
                    output, loss = run_epoch(data_generator, self.model, self.criterion)
                    if epoch % 10 ==0 and self.rank==1:
                        o = torch.topk(output[11].softmax(-1), 5, -1) 
                        write((o.indices.cpu().detach().numpy().tolist(),
                            o.values.cpu().detach().numpy().tolist() ), '/home/fangyikai/yanti/spec2mol/out.txt')
                    batch_size = data_generator.batch_size
                if binary:
                    output = torch.sigmoid(output)
                # loss = self.criterion(output, target.to(self.model.device))
                losses.update(loss.item(), batch_size)
                output = output.detach().cpu().numpy()
                
                prediction = []
                if self.pretrain:
                    if self.rank==0:
                        bar.set_description(f'Epoch{epoch:3d}, valid loss:{losses.avg:6f}')
                else:
                    flag = True
                    # if target.ndim == 1:
                    #     prediction = torch.argmax(output, dim=1).view(-1).detach().cpu().numpy()
                    #     accuracy = (prediction == target).mean()
                    #     accuracies.update(accuracy.item(), data.size(0))
                    #     bar.set_description(
                    #         f'Epoch{epoch:3d}, valid loss:{losses.avg:6f} , accuracy:{accuracies.avg:6f}')
                    # elif binary and len(set(target[:, 0])) == 2:
                    #     prediction = torch.greater_equal(output, 0.5).to(torch.float64).cpu().detach().numpy()
                    #     accuracy = metrics.f1_score(target, prediction, average='macro', zero_division=0.0)
                    #     accuracies.update(accuracy, data.size(0)) # accuracy.item()
                    #     bar.set_description(
                    #         f'Epoch{epoch:3d}, valid loss:{losses.avg:6f} , valid accuracy:{accuracies.avg:6f}')
                    # else:
                        # src_mask = Variable(torch.ones(1, 1, self.src_length))
                        # accuracy = (output == target).mean()
                        # accuracy = metrics.accuracy_score(target, prediction, zero_division=0.0)
                        # out_freq = None
                        # if epoch % 20 == 0:
                    if epoch % 5 == 0:
                        output = greedy_decode(self.model, data_generator, self.src_length, self.max_len, sample_num=5)
                        for out in output:
                            prediction+=out
                        target = data_generator.true_seq
                        # torch.save((output, target), '/home/fangyikai/yanti/spec2mol/out.pt')
                        token_acc = eval_tokens(output, target) / np.sum([len(i) for i in target])
                        accuracies.update(token_acc, data_generator.batch_size) # accuracy.item()
                        if self.rank==0:
                            bar.set_description(f'Epoch{epoch:3d}, valid loss:{losses.avg:6f} , valid accuracy:{accuracies.avg:6f}')  
                    else:
                        if self.rank==0:
                            bar.set_description(
                                f'Epoch{epoch:3d}, valid loss:{losses.avg:6f}')                        
        self.lr_scheduler.step()
        # if target.ndim == 1:
        #     logging.info(f'Epoch{epoch:3d}, valid loss:{losses.avg:6f}, valid accuracy:{accuracies.avg:6f}')
        # else:
        #     logging.info(f'Epoch{epoch:3d}, valid loss:{losses.avg:6f}')
        return losses.avg, accuracies.avg, prediction if flag else None

    def test_epoch(self, binary=False):
        accuracies = AverageMeter()
        losses = AverageMeter()
        self.model.eval()
        bar = tqdm(self.test_loader, ncols=100)
        outputs = []
        predicted = []
        true = []
        accuracy = 0
        with torch.no_grad():
            for _, dataset in enumerate(self.test_loader):
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
                    prediction = greedy_decode(self.model, data_generator, self.src_length, self.max_len)
                    target = data_generator.true_seq
                    result = eval_canonical_smiles(prediction, target)
                    accuracy = result[0] / len(prediction)
                accuracies.update(accuracy, len(prediction))
                bar.set_description(
                    f"test loss: {losses.avg:.5f} accuracy:{accuracies.avg:.5f}")
                predicted += prediction.tolist() if type(prediction) != list else prediction
                true += target.tolist() if type(target) != list else target
        return outputs, predicted, true, accuracies.avg, losses.avg


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
        criterion=torch.nn.CrossEntropyLoss(reduction='none', ignore_index=1)
        
    optimizer = torch.optim.AdamW(
        model.parameters(), **kwargs['Adam_params'])
    if kwargs['checkpoint']:
        load_optm_state(optimizer, kwargs['checkpoint']['optimizer_state'])
        
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
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
            val_loss_global = es(val_loss_global, f'{save_path}/{epoch}_vloss_{str(val_loss)[2:6]}.pth', model, optimizer, lr_scheduler)
        elif acc:
            val_acc_global = es(acc, f'{save_path}/{epoch}_f1_{str(acc)[2:6]}.pth', model, optimizer, lr_scheduler)
        else:
            assert ValueError('not metrics')
        if es.early_stop == 1:
            break
        if kwargs['rank'] == 0:
            writer.add_scalar("Loss/train", train_loss_global, epoch)
            writer.add_scalar("Loss/val", val_loss_global, epoch)
            if val_acc_global:
                writer.add_scalar("Accuracy/val", val_acc_global, epoch)
            if seq_pred:
                writer.add_histogram('Token Frequency/val', torch.stack(seq_pred), epoch, bins=1)
        dist.barrier()
    print(es.val_score)
    cleanup()


def test_model(model, ds, device='cpu', verbose=True, **kwargs):
    
    test_loader = make_testloader(ds)
    binary = False#True if 'ir' in ds or 'raman' in ds else 
    criterion = torch.nn.BCELoss() if binary else torch.nn.CrossEntropyLoss(reduction='none', ignore_index=1)
    engine = Engine(test_loader=test_loader,
                    criterion=criterion, model=model, device=device, **kwargs)
    outputs, pred, true, _, _ = engine.test_epoch(binary=binary)

    if verbose:
        # print(metrics.classification_report(true, pred, digits=4))
        smiles_result = eval_canonical_smiles(pred, true)
        print('True SMILES: %.4f\n' %(smiles_result[0].item()/len(pred)))
        print('False SMILES: %.4f\n' %(smiles_result[1].item()/len(pred)))
        print('Invalid SMILES: %.4f\n' %(smiles_result[2].item()/len(pred)))
        print('Token accuracy: %.4f\n' %(eval_tokens(pred, true)/len(pred)))
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