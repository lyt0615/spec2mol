"""
@File        :main.py
@Author      :Yanti Liu
@EMail       :aandytliu@gmail.com
"""

import os, time, logging, argparse, config, json
from utils.utils import seed_everything, load_net_state, train_model, test_model, cleanup, get_seq
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_args_parser():
    parser = argparse.ArgumentParser('---', add_help=False)

    # basic params
    parser.add_argument('--net', default='Transformer',
                        help="Choose network")
    parser.add_argument('--ds', default='raman2mol',
                        help="Choose dataset")
    parser.add_argument('--gpu_ids', default=None,
                        help="Indices of GPUs to be used")
    parser.add_argument('--device', default='cuda:0',
                        help="Choose GPU device")
    parser.add_argument('--mode', default='train',
                        help="choose mode: 'train', 'test', 'pretrain_mol' or 'pretrain_spec'")
    parser.add_argument('-debug', '--debug', action='store_true',
                        help="start debug")
    parser.add_argument('--base_checkpoint',
                        help="Choose base model for fine-tune")
    parser.add_argument('--test_checkpoint',
                        help="Choose checkpoint for test")
    parser.add_argument('--base_model_path', default=None,
                        help="Choose checkpoint for reloading or finetuning")
    parser.add_argument('--seed',
                        default=2024,
                        help="Random seed")
    parser.add_argument('--max_len',
                        default=40,
                        help="Maxial length of output sequence when decoding")
    parser.add_argument('--depth', default=4,
                        help="Transformer layers")  
    parser.add_argument('--d_model', default=512,
                        help="Transformer hidden dimension")  
    parser.add_argument('--n_heads', default=8,
                        help="Number of attention heads")      
    parser.add_argument('--to_smarts', action='store_true',
                        help="Generating SMARTS")    
    parser.add_argument('--epoch',
                        help="epochs for training")
    parser.add_argument('--ddp', default=False,
                        help="Using DDP")
    args = parser.parse_args()
    return args

def catch_exception(ds, net_, ts, mode):
    import traceback
    import shutil

    traceback.print_exc()
    
    if os.path.exists(f'logs/{ds}/{net_}/{ts}_{mode}.log'):
        os.remove(f'logs/{ds}/{net_}/{ts}_{mode}.log') 
        print('unexpected log has been deleted')
    if os.path.exists(f'checkpoints/{ds}/{net_}/{ts}'):
        shutil.rmtree(f'checkpoints/{ds}/{net_}/{ts}')
        print('unexpected tensorboard record has been deleted')


def main(rank=None, world_size=1, gpu_ids=None):
    import torch
    args = get_args_parser()
    mode = args.mode
    seed_everything(int(args.seed))
    params = {'net': config.NET, 'strategy': config.STRATEGY['pretrain'] if 'pretrain' in args.mode else config.STRATEGY[mode]}
    if not mode == 'test':
        params['strategy']['world_size'] = world_size
        params['strategy']['rank'] = rank
        params['strategy']['gpu_ids'] = gpu_ids
    params['strategy']['mode'] = mode
    params['strategy']['checkpoint'] = torch.load(args.base_model_path, 
                                                  map_location={'cuda:%d' % 0: 'cuda:%d' % rank},
                                                  weights_only=True) if args.base_model_path else None
    params['strategy']['to_smarts'] = args.to_smarts
    ts = time.strftime('%Y-%m-%d_%H_%M', time.localtime())
    ds = args.ds
    net_ = args.net
    device = args.device

    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists(f'logs/{ds}'):
        os.mkdir(f'logs/{ds}')
    if not os.path.exists(f'logs/{ds}/{net_}'):
        os.mkdir(f'logs/{ds}/{net_}')


    if mode == 'train':
        if not os.path.exists(f"checkpoints/{ds}"):
            os.mkdir(f"checkpoints/{ds}")
        if not os.path.exists(f"checkpoints/{ds}/{net_}"):
            os.mkdir(f"checkpoints/{ds}/{net_}")
        if not os.path.exists(f"checkpoints/{ds}/{net_}/{ts}"):
            os.mkdir(f"checkpoints/{ds}/{net_}/{ts}")
    elif FileExistsError: pass

    if mode == 'train' or mode == 'test':
        with open('models/moltokenizer/vocab.json', encoding="utf-8") as f:
            vocab = json.load(f)
        tgt_vocab = len(vocab)
    else:
        tgt_vocab = 1024
    params['net']['tgt_vocab'] = tgt_vocab

    from models.Transformer import make_model
    N, d_model, h = int(args.depth), int(args.d_model), int(args.n_heads)
    net, src_length = make_model(tgt_vocab, N=N, d_model=d_model, h=h, mode=mode)
    params['strategy']['src_length'] = src_length
    
    if rank == 0:
        logging.basicConfig(
            filename=f'logs/{ds}/{net_}/{ts}_{mode}.log',
            format='%(levelname)s:%(message)s',
            level=logging.INFO,
        )

        logging.info({k: v for k, v in args.__dict__.items() if v})
        save_path = f"checkpoints/{ds}/{net_}/{ts}"
        logging.info(net)
        logging.info(params)
    
    try:
        if mode == 'train' or args.debug or 'pretrain' in mode:
            if args.base_model_path:
                net = load_net_state(net, params['strategy']['checkpoint']['model_state'])
            pretrain = True if 'pretrain' in mode else False
            train_model(net, save_path, ds=args.ds, **params['strategy'], pretrain=pretrain)

        elif mode == 'tune':
            import torch
            base_model_path = args.base_model_path
            print(base_model_path)
            net = load_net_state(net, torch.load(f'{base_model_path}.pth', map_location={'cuda:%d' % 0: 'cuda:%d' % rank}, weights_only=True))
            train_model(net, save_path=f"{base_model_path}/tune", ds=args.ds, **params['strategy'])

        elif mode == 'test':
            import torch
            test_model_path = args.test_checkpoint
            print(test_model_path)
            net = load_net_state(net, torch.load(test_model_path,
                                                map_location={'cuda:0': device, 'cuda:1': device}, weights_only=True)['model_state'])
            _, pred, true = test_model(net, device=device, ds=args.ds, **params['strategy'])
            logging.info('Prediction, Target')
            for p, t in zip(pred, true):
                logging.info(f'{p}   {t}\n')

    except Exception as e:
        import traceback 
        traceback.print_exc()
        catch_exception(ds, net_, ts, mode)

if __name__ == "__main__":

    if not get_args_parser().mode == 'test':
        gpu_ids = get_args_parser().gpu_ids
        if '-' in gpu_ids:
            g1, g2 = int(gpu_ids.split('-')[0]), int(gpu_ids.split('-')[1])
            gpu_ids = str(g1)
            for i in range(g1+1, g2+1):
                gpu_ids = gpu_ids + ',' + str(i)
        world_size = len(gpu_ids.split(','))       
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        mp.spawn(main, args=(world_size, gpu_ids), nprocs=world_size)
    else:
        main()