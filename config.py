NET = {'conv_ksize':3, 
       'conv_padding':1, 
       'conv_init_dim':32, 
       'conv_final_dim':256, 
       'conv_num_layers':4, 
       'mp_ksize':2, 
       'mp_stride':2, 
       'fc_dim':1024, 
       'fc_num_layers':3, 
       'mixer_num_layers':4,
       'tgt_vocab':957,
       'use_mixer':True
       }


STRATEGY = {
    'train': {
        "batch_size": 8,
        "epoch": 1000,
        "patience": 200,
        'train_size': None,
        "optmizer": "Adam",
        "Adam_params": {"lr": 1e-4}, # for qm9s

    },
    'tune': {
        "batch_size": 64,
        # "batch_size": 8, # for Bacteria
        "epoch": 600,
        "patience": 50,
        'train_size': None,
        "optmizer": "Adam",
        "Adam_params": {"lr": 1e-5},
    },
    'pretrain': {
        "batch_size": 64,
        "epoch": 200,
        "patience": 50,
        'train_size': 0.85,
        "optmizer": "Adam",
        "Adam_params": {"lr": 1e-5},
    }
}

tokens = ['<pad>', '<bos>', 'Cl','Br','C','c','O','o','N','n','S','s','F','I','P','(',')','=','#','1','2','3','@', '<eos>']
