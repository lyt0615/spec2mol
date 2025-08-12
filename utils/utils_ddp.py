import logging
from tqdm import tqdm
import numpy as np
import random
import os
import torch
from sklearn import metrics
from utils.dataloader_ddp import make_trainloader, make_testloader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group
import os
from models.Transformer_modules import run_epoch, LossCompute, greedy_decode, LabelSmoothing, Collator, Variable


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    try:
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
    except RuntimeError:
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
        init_process_group(backend="gloo", rank=rank, world_size=world_size, 
                      init_method="env://?use_libuv=False") # disable libuv

   
def cleanup():
    dist.destroy_process_group()
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def get_vib_masked(spec,
                mask_ratio=0.15,
                # silent_start=1800,   # cm⁻¹
                # silent_end=2800,
                # wavenumbers=torch.linspace(4000,400,1024),    # 1-D tensor length L
                mask_value=0.0
                ):
    B, *_ = spec.shape
    len_keep = int((1 - mask_ratio) * spec[0].numel())
    noise = torch.rand(B, spec[0].numel(), device=spec.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    mask = torch.ones_like(spec.view(B, -1))
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore).view_as(spec).bool()

    masked_spec = spec.masked_fill(mask, mask_value)
        # mask = mask.view_as(spec)
    return mask, masked_spec

    
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
    def __init__(self, patience=10, mode='max', delta=0.0001, rank=0):
        self.patientce = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.rank = rank
        if self.mode == 'min':
            self.val_score = np.inf
        else:
            self.val_score = -np.inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == 'min':
            score = -1. * epoch_score
        else:
            score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path, self.rank)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patientce:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path, self.rank)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path, rank):
        if rank == 0:
            torch.save(model.state_dict(), model_path)
            self.val_score = epoch_score
        else: pass
        dist.barrier()


class Engine:
    def __init__(self, train_loader=None, val_loader=None, test_loader=None,
                 criterion=None, optimizer=None, scheduler=None, device='cpu',
                 model=None, **kwargs):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.src_length = kwargs['src_length']
        self.max_len = kwargs['max_len']
        mode = kwargs['mode']
        self.pretrain = 1 if mode=='pretrain' else 0
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        if nn.BatchNorm1d in list(model.modules()):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
        rank = kwargs['rank']
        if not mode=='test':
            self.model = DDP(model.to(rank), device_ids=[rank], find_unused_parameters=False) # model
            self.device = self.model.device
        else: 
            self.device = device
            self.model=model.to(device)
        # print(sum(param.untyped_storage().nbytes() for param in self.model.parameters()))
    def train_epoch(self, epoch, binary=False):

        losses = AverageMeter()
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)
        bar = tqdm(self.train_loader, ncols=100)
        
        for _, dataset in enumerate(bar):
            self.optimizer.zero_grad()
            if self.pretrain:
                mask, data = get_vib_masked(dataset[0])
                target = dataset[0][mask]
                output = self.model(dataset[0]).unsqueeze(1)[mask]
            else: 
                
                # data, target = dataset
                data_generator = Collator(dataset, self.src_length, self.device)
                output, loss = run_epoch(data_generator, self.model,
                    LossCompute(self.model, self.criterion, None))
                loss.backward()
            if binary:
                output = torch.sigmoid(output)
            # loss = self.criterion(output, target.to(self.model.device))
            # loss.backward()
            self.optimizer.step()

            losses.update(loss.data.item(), data_generator.batch_size)
            bar.set_description(
                f'Epoch{epoch:3d}, train loss:{losses.avg:6f}')

        logging.info(f'Epoch{epoch:3d}, train loss:{losses.avg:6f}')
        return losses.avg

    def evaluate_epoch(self, epoch, binary=False):

        accuracies = AverageMeter()
        losses = AverageMeter()
        flag = True
        self.model.eval()
        bar = tqdm(self.val_loader, ncols=100)
        with torch.no_grad():
            for _, dataset in enumerate(bar):
                if self.pretrain:
                    mask, data = get_vib_masked(dataset[0])
                    target = dataset[0][mask]
                    output = self.model(dataset[0]).unsqueeze(1)[mask]
                else: 
                    # data, target = dataset
                    data_generator = Collator(dataset, self.src_length, self.device)
                    target = data_generator.tgt_y
                    output, loss = run_epoch(data_generator, self.model,
                                LossCompute(self.model, self.criterion, None))
                    prediction = greedy_decode(self.model, data_generator, self.max_len)
                if binary:
                    output = torch.sigmoid(output)
                # loss = self.criterion(output, target.to(self.model.device))
                losses.update(loss.item(), data_generator.batch_size)
                output = output.detach().cpu().numpy()
                
                if self.pretrain:
                    bar.set_description(
                    f'Epoch{epoch:3d}, valid loss:{losses.avg:6f}')
                else:
                    flag = True
                    if target.ndim == 1:
                        prediction = torch.argmax(output, dim=1).view(-1).detach().cpu().numpy()
                        accuracy = (prediction == target).mean()
                        accuracies.update(accuracy.item(), data.size(0))
                        bar.set_description(
                            f'Epoch{epoch:3d}, valid loss:{losses.avg:6f} , accuracy:{accuracies.avg:6f}')
                    elif binary and len(set(target[:, 0])) == 2:
                        prediction = torch.greater_equal(output, 0.5).to(torch.float64).cpu().detach().numpy()
                        accuracy = metrics.f1_score(target, prediction, average='macro', zero_division=0.0)
                        accuracies.update(accuracy, data.size(0)) # accuracy.item()
                        bar.set_description(
                            f'Epoch{epoch:3d}, valid loss:{losses.avg:6f} , valid accuracy:{accuracies.avg:6f}')
                    else:
                        # src_mask = Variable(torch.ones(1, 1, self.src_length))
                        # accuracy = (output == target).mean()
                        accuracy = metrics.accuracy_score(target, prediction, zero_division=0.0)
                        accuracies.update(accuracy, data_generator.batch_size) # accuracy.item()
                        bar.set_description(
                            f'Epoch{epoch:3d}, valid loss:{losses.avg:6f} , valid accuracy:{accuracies.avg:6f}')                        


        if self.scheduler:
            self.scheduler.step()
        if target.ndim == 1:
            logging.info(f'Epoch{epoch:3d}, valid loss:{losses.avg:6f}, valid accuracy:{accuracies.avg:6f}')
        else:
            logging.info(f'Epoch{epoch:3d}, valid loss:{losses.avg:6f}')

        return losses.avg, accuracies.avg if flag else None

    def test_epoch(self, binary=False):

        accuracies = AverageMeter()
        losses = AverageMeter()

        self.model.eval()
        bar = tqdm(self.test_loader, ncols=100)
        outputs = []
        predicted = []
        true = []
        with torch.no_grad():
            for _, dataset in enumerate(bar):
                data, target = data.to(self.device), target.to(self.device)
                data_generator = Collator(dataset, self.src_length, self.device)
                if binary:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    target = target.detach().cpu().numpy()
                    output = torch.sigmoid(output)
                    prediction = torch.greater_equal(output, 0.5).to(torch.float64).cpu().detach().numpy()
                    accuracy = metrics.f1_score(target, prediction, 
                                                average='macro', zero_division=0.0)
                else:
                    # src_mask = Variable(torch.ones(1, 1, self.src_length))
                    prediction = greedy_decode(self.model, data_generator, self.max_len)
                    accuracy = metrics.precision_score(target, output, average='micro', zero_division=0.0)

                outputs.append(output.detach().cpu().numpy())
                losses.update(loss.item(), data.size(0))
                accuracies.update(accuracy, data.size(0))
                bar.set_description(
                    f"test loss: {losses.avg:.5f} accuracy:{accuracies.avg:.5f}")
                predicted += prediction.tolist()
                true += target.tolist()
            outputs = np.concatenate(outputs)
        return outputs, np.array(predicted), np.array(true), accuracies.avg, losses.avg


def load_net_state(net, state_dict):
    # check the keys and load the weight
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


def train_model(model, save_path, ds, **kwargs):
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir = save_path)
    
    ddp_setup(kwargs['rank'], kwargs['world_size'])
    train_loader, val_loader = make_trainloader(
        ds, batch_size=kwargs['batch_size'],
        num_workers=1, train_size=kwargs['train_size'],
        seed=42, mode=kwargs['mode'], )

    test_loader = make_testloader(ds, ) if not kwargs['mode']=='pretrain' else None
    binary = False  #True if 'ir' in ds or 'raman' in ds else 
    if kwargs['mode'] == 'pretrain':
        criterion = nn.MSELoss()
    elif binary:
        criterion = torch.nn.BCELoss()
    else:
        criterion = LabelSmoothing(size=11, padding_idx=0, 
                                   smoothing=0.0, criterion=torch.nn.CrossEntropyLoss(reduction='none'))
    optimizer = torch.optim.AdamW(
        model.parameters(), **kwargs['Adam_params'])
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    mode = 'max' if not kwargs['mode'] == 'pretrain' else 'min'
    es = EarlyStop(patience=kwargs['patience'], mode=mode, rank=kwargs['rank'])
    engine = Engine(train_loader=train_loader, val_loader=val_loader,
                    test_loader=test_loader,criterion=criterion, optimizer=optimizer, 
                    scheduler=scheduler, model=model, **kwargs)
    # start to train 
    for epoch in range(kwargs['epoch']):
        train_loss = engine.train_epoch(epoch, binary=binary)
        val_loss, acc = engine.evaluate_epoch(epoch, binary=binary)
        # _, _, _, test_acc, test_loss = engine.test_epoch(binary=binary)
        if kwargs['mode']=='pretrain':
            es(val_loss, model, f'{save_path}/{epoch}_vloss_{str(val_loss)[2:6]}.pth')
        elif acc:
            es(acc, model, f'{save_path}/{epoch}_f1_{str(acc)[2:6]}.pth')
        else:
            assert ValueError('not metrics')
        if es.early_stop:
            break
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", acc, epoch)
        # writer.add_scalar("Loss/test", test_loss, epoch)
        # writer.add_scalar("Accuracy/test", test_acc, epoch)
    cleanup()
    print(es.val_score)


def test_model(model, ds, device='cpu', verbose=True, **kwargs):
    
    test_loader = make_testloader(ds, )
    binary = False#True if 'ir' in ds or 'raman' in ds else 
    criterion = torch.nn.BCELoss() if binary else torch.nn.CrossEntropyLoss(reduction='none')
    engine = Engine(test_loader=test_loader,
                    criterion=criterion, model=model, device=device, **kwargs)
    outputs, pred, true, _, _ = engine.test_epoch(binary=binary)

    if verbose:
        # print(metrics.classification_report(true, pred, digits=4))
        print('Exact match rate (EMR): %.4f\n' %metrics.accuracy_score(true, pred))
        # logging.info(metrics.classification_report(true, pred, digits=4))
        # logging.info(f'accuracy:{metrics.accuracy_score(true, pred):.5f}')
        logging.info('Exact match rate (EMR): %.4f\n' %metrics.accuracy_score(true, pred))
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