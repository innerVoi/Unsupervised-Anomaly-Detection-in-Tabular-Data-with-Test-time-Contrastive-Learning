import torch
import torch.optim as optim
from Data_Prepare.DataLoader import get_dataloader
from Model.Model import CDTL
from Model.Loss import LossFunction
from Model.Score import ScoreFunction
from utils import aucPerformance, get_logger, F1Performance
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.neighbors import NearestNeighbors


class Trainer(object):
    def __init__(self, run: int, model_config: dict):
        self.run = run
        self.sche_gamma = model_config['sche_gamma']
        self.device = model_config['device']
        self.learning_rate = model_config['learning_rate']
        self.batch_size = model_config['batch_size']
        self.num_workers = model_config['num_workers']
        self.model = CDTL(model_config).to(self.device)
        self.loss_fuc = LossFunction(model_config).to(self.device)
        self.score_func = ScoreFunction(model_config).to(self.device)
        self.train_loader, self.test_loader, self.train_set, self.test_set = get_dataloader(model_config)

        self.input_dim = model_config['data_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.gamma = model_config['gamma']
        self.dataset_name = model_config['dataset_name']


    def joint_training(self, epochs):
        train_logger = get_logger('train_log.log')
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")
        min_loss = float(np.inf)
        for epoch in range(epochs):
            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                x_pred, z, masks, z_x, z_x_pred = self.model(x_input)
                loss, mse, divloss, ssl_loss = self.loss_fuc(x_input, x_pred, masks, z_x, z_x_pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t mse={:.4f}\t divloss={:.4f}\t ssl_loss={:.4f}'
            train_logger.info(info.format(epoch,loss.cpu(),mse.cpu(),divloss.cpu(),ssl_loss.cpu()))
            if loss < min_loss:
                torch.save(self.model, f'./saved_model/{self.dataset_name}.pth')
                min_loss = loss
        print("Training complete.")
        train_logger.handlers.clear()

    def evaluate(self, mse_rauc, mse_ap, mse_f1):
        model = torch.load(f'./saved_model/{self.dataset_name}.pth')
        model.eval()
        mse_score, test_label = [], []
        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)
            x_pred, z, masks, z_x, z_x_pred = model(x_input)
            mse_batch = self.score_func(x_input, x_pred)
            mse_batch = mse_batch.data.cpu()
            mse_score.append(mse_batch)
            test_label.append(y_label)
        mse_score = torch.cat(mse_score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        mse_rauc[self.run], mse_ap[self.run] = aucPerformance(mse_score, test_label)
        mse_f1[self.run] = F1Performance(mse_score, test_label)

        return mse_score, test_label, self.train_set, self.test_set

    def TTCL(self, mse_rauc, mse_ap, mse_f1, error_score, train_data, test_data, K, normal_tuning_epochs, abnormal_tuning_epochs):

        model = torch.load(f'./saved_model/{self.dataset_name}.pth')
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)

        # 根据测试集中的正常/异常总数量，设置每次基于置信度选择的正常/异常数量
        normal_num = (test_data.targets == 0).sum()
        abnormal_num = (test_data.targets == 1).sum()
        ratio = int(normal_num/abnormal_num)
        select_abnormal_num = torch.div(abnormal_num, 10, rounding_mode='floor')
        select_normal_num = int(ratio*select_abnormal_num)+1
        print(f'In each iteration, select_abnormal_num={select_abnormal_num}, select_normal_num={select_normal_num}')

        # 设置每次选择的样本数量, 当剩余样本数量不足以执行下一轮选择时，对比调整停止
        target_size = select_abnormal_num + select_normal_num
        score = np.squeeze(error_score).tolist()
        remain_index = [i for i in range(len(score))]

        def iter_optim(score, remain_index, train_data, test_data):
            # 1. 获得之前正常的嵌入表示, 用knn拟合模型
            embeddings_list = []
            train_data_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
            for step, (x_input, _) in enumerate(train_data_loader):
                x_input = x_input.to(self.device)
                x_pred, z, masks, z_x, z_x_pred = model(x_input)
                embeddings_list.append(z.cpu())
            embeddings = torch.cat(embeddings_list)
            embeddings_np = embeddings.detach().numpy().reshape(len(train_data), -1)

            nn_model = NearestNeighbors(n_neighbors=K, algorithm='auto')
            nn_model.fit(embeddings_np)

            # 2. 根据重构损失得到的异常概率，选择本轮待处理的正常和异常
            min_score, max_score = min(score), max(score)
            norm_score = [(score[i] - min_score) / (max_score - min_score) for i in range(len(score))]
            sorted_norm_score_index = np.argsort(norm_score)
            print(f'remain_index len {len(remain_index)}, norm_score len {len(norm_score)}')

            # 列出本轮选择的正常/异常索引
            input_normal_index = [remain_index[i] for i in sorted_norm_score_index[:select_normal_num]]
            input_abnormal_index = [remain_index[i] for i in sorted_norm_score_index[-select_abnormal_num:]]
            input_index = input_normal_index + input_abnormal_index
            input_label = [0] * select_normal_num + [1] * select_abnormal_num
            remain_index = list(set(remain_index) - set(input_index))

            new_test_normal = Subset(test_data, input_normal_index)
            new_test_abnormal = Subset(test_data, input_abnormal_index)
            new_data = ConcatDataset([train_data, new_test_normal])
            new_test_normal_loader = DataLoader(new_test_normal, batch_size=self.batch_size, shuffle=False)
            new_test_abnormal_loader = DataLoader(new_test_abnormal, batch_size=self.batch_size, shuffle=True)
            new_data_loader = DataLoader(new_data, batch_size=self.batch_size, shuffle=True)

            for epoch in range(normal_tuning_epochs):
                for step, (x_input, _) in enumerate(new_data_loader):
                    x_input = x_input.to(self.device)
                    x_pred, z, masks, z_x, z_x_pred = model(x_input)
                    loss, mse, divloss, ssl_loss = self.loss_fuc(x_input, x_pred, masks, z_x, z_x_pred)
                    z_normal = z.cpu().detach().numpy().reshape(x_input.size(0), -1)
                    distances, indices = nn_model.kneighbors(z_normal)
                    nearest_embeddings = embeddings_np[indices]

                    # 计算对比损失，让正常样本接近K近邻正常样本
                    z_normal_expand = z.reshape(x_input.size(0), -1).unsqueeze(1).cpu().detach().numpy()
                    nearest_embeddings = torch.tensor(nearest_embeddings).to(self.device)
                    z_normal_expand = torch.tensor(z_normal_expand).to(self.device)
                    contrastive_loss = torch.norm(z_normal_expand - nearest_embeddings, dim=2).mean()

                    final_loss = loss + contrastive_loss
                    optimizer.zero_grad()
                    final_loss.backward()
                    optimizer.step()
                scheduler.step()

            for epoch in range(abnormal_tuning_epochs):
                for step, (x_input, _) in enumerate(new_test_abnormal_loader):
                    x_input = x_input.to(self.device)
                    x_pred, z, masks, z_x, z_x_pred = model(x_input)
                    loss, mse, divloss, ssl_loss = self.loss_fuc(x_input, x_pred, masks, z_x, z_x_pred)
                    z_abnormal = z.cpu().detach().numpy().reshape(x_input.size(0), -1)
                    distances, indices = nn_model.kneighbors(z_abnormal)
                    nearest_embeddings = embeddings_np[indices]

                    z_abnormal_expand = z.reshape(x_input.size(0), -1).unsqueeze(1).cpu().detach().numpy()
                    nearest_embeddings = torch.tensor(nearest_embeddings).to(self.device)
                    z_abnormal_expand = torch.tensor(z_abnormal_expand).to(self.device)
                    contrastive_loss = torch.norm(z_abnormal_expand - nearest_embeddings, dim=2).mean()

                    final_loss = - (loss + contrastive_loss)
                    optimizer.zero_grad()
                    final_loss.backward()
                    optimizer.step()
                scheduler.step()


            if len(remain_index) > target_size:
                remain_data = Subset(test_data, remain_index)
                remain_data_loader = DataLoader(remain_data, batch_size=self.batch_size, shuffle=False)
                mse_score, test_label = [], []
                for step, (x_input, y_label) in enumerate(remain_data_loader):
                    x_input = x_input.to(self.device)
                    x_pred, z, masks, z_x, z_x_pred = model(x_input)
                    mse_batch = self.score_func(x_input, x_pred)
                    mse_batch = mse_batch.data.cpu()
                    mse_score.append(mse_batch)
                    test_label.append(y_label)
                mse_score = torch.cat(mse_score, axis=0).numpy()
                score = np.squeeze(mse_score).tolist()

                train_data = new_data
                iter_optim(score, remain_index, train_data, test_data)

        iter_optim(score, remain_index, train_data, test_data)

        model.eval()
        mse_score, test_label = [], []
        embeddings_list = []
        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)
            x_pred, z, masks, z_x, z_x_pred = model(x_input)
            mse_batch = self.score_func(x_input, x_pred)
            mse_batch = mse_batch.data.cpu()
            mse_score.append(mse_batch)
            test_label.append(y_label)
            embeddings_list.append(z.cpu())

        embeddings = torch.cat(embeddings_list)
        torch.save(embeddings, f'embeddings/{self.dataset_name}-2.pt')

        mse_score = torch.cat(mse_score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        mse_rauc[self.run], mse_ap[self.run] = aucPerformance(mse_score, test_label)
        mse_f1[self.run] = F1Performance(mse_score, test_label)

        return mse_score, test_label, self.train_set, self.test_set

