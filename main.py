import torch
import numpy as np
from Model.Trainer import Trainer

model_config = {

    'dataset_name': 'thyroid',
    'data_dim': 6,
    'epochs': 200,
    'learning_rate': 0.005,
    'sche_gamma': 0.98,
    'mask_num': 15,
    'lambda': 0,
    'gamma': 1,
    'K': 3,
    'normal_tuning_epochs': 10,
    'abnormal_tuning_epochs': 5,

    'device': 'cuda:0',
    'data_dir': 'Data/',
    'runs': 1,
    'num_workers': 0,
    'batch_size': 512,
    'en_nlayers': 3,
    'de_nlayers': 3,
    'hidden_dim': 256,
    'z_dim': 128,
    'mask_nlayers': 3,
    'random_seed': 42,

}

if __name__ == "__main__":
    # 随机种子设置
    torch.manual_seed(model_config['random_seed'])
    torch.cuda.manual_seed(model_config['random_seed'])
    np.random.seed(model_config['random_seed'])
    if model_config['num_workers'] > 0:
        torch.multiprocessing.set_start_method('spawn')

    result = []
    runs = model_config['runs']
    mse_rauc, mse_ap, mse_f1 = np.zeros(runs), np.zeros(runs), np.zeros(runs)


    for i in range(runs):
        trainer = Trainer(run=i, model_config=model_config)
        trainer.joint_training(model_config['epochs'])
        mse_score, test_label, train_set, test_set = trainer.evaluate(mse_rauc, mse_ap, mse_f1)
        mse_score, test_label, train_set, test_set, = trainer.TTCL(mse_rauc, mse_ap, mse_f1, mse_score,
                                                                               train_set, test_set, model_config['K'],
                                                                               model_config['normal_tuning_epochs'],
                                                                               model_config['abnormal_tuning_epochs'])
        mean_mse_auc, mean_mse_pr, mean_mse_f1 = np.mean(mse_rauc), np.mean(mse_ap), np.mean(mse_f1)
        print("mse: average AUC-ROC: %.4f  average AUC-PR: %.4f  average f1: %.4f" % (
        mean_mse_auc, mean_mse_pr, mean_mse_f1))
