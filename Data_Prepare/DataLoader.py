from torch.utils.data import DataLoader
from Data_Prepare.MyDataset import MatDataset, NpzDataset


def get_dataloader(model_config: dict):
    dataset_name = model_config['dataset_name']

    if dataset_name in ['arrhythmia', 'breastw', 'cardio', 'glass', 'ionosphere', 'mammography', 'pima', 'satellite', 'satimage-2', 'thyroid', 'wbc']:
        train_set = MatDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='train')
        test_set = MatDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='eval')

    elif dataset_name in ['wine', 'optdigits', 'pendigits', 'cardiotocography']:
        train_set = NpzDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='train')
        test_set = NpzDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='eval')

    else:
        print('Dataset Error!')

    train_loader = DataLoader(train_set,
                              batch_size=model_config['batch_size'],
                              num_workers=model_config['num_workers'],
                              shuffle=False)
    test_loader = DataLoader(test_set, batch_size=model_config['batch_size'], shuffle=False)
    return train_loader, test_loader, train_set, test_set