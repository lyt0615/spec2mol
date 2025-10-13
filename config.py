NET = { 
       'tgt_vocab':181,
       }


STRATEGY = {
    'train': {
        "batch_size": 64,
        "epoch": 2000,
        "patience": 500,
        'train_size': None,
        "optmizer": "Adam",
        "Adam_params": {"lr": 6e-4},
    },
    'tune': {
        "batch_size": 64,
        "epoch": 2000,
        "patience": 50,
        'train_size': None,
        "optmizer": "Adam",
        "Adam_params": {"lr": 1e-5},
        'greedy_decode':{
        'max_len': 30,
        'repetition_penalty':1
        }
    },
    'pretrain': {
        "batch_size": 64,
        "epoch": 600,
        "patience": 50,
        'train_size': 0.85,
        "optmizer": "Adam",
        "Adam_params": {"lr": 1e-5},
    },
    'test': {
        "batch_size":2,
        'greedy_decode':{
            'max_len': 30,
            'repetition_penalty':1.9
        }
    }
}

tokens = ['<pad>', '<bos>', '<eos>', 'Cl','Br','C','c','O','o','N','n','S','s','F','I','P','(',')','=','#','1','2','3','@']
