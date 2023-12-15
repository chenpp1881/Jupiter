import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
from itertools import cycle
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from clip_mlm import CLIP as CLIPMLM
import matplotlib.pyplot as plt
import torch.nn.functional as F
from data_utils import data_package
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

logger = logging.getLogger(__name__)


def all_metrics(y_true, y_pred, is_training=False):
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training

    return f1.item(), precision.item(), recall.item(), tp.item(), tn.item(), fp.item(), fn.item()


def get_last_resume_file(args):
    files = os.listdir(args.savepath + '/' + args.dataset)
    files = [file for file in files if file.split('.')[-1] == 'pth']
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(files) == 0:
        return None
    return os.path.join(args.savepath, args.dataset, files[-1])


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        return torch.mean((label) * torch.pow(euclidean_distance, 2) +
                          (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                  2))


class ClipTrainer():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.start_epoch = 0
        self.k = 0

        self.text_seq_len = args.max_length
        clip = CLIPMLM(
            args=args,
            dim_text=768,
            text_seq_len=self.text_seq_len,
            model_name=args.model_path
        )
        self.optimizer = optim.AdamW(clip.parameters(), lr=args.lr_clip)
        self.c_loss = ContrastiveLoss()

        self.cla_optimizer = optim.AdamW(clip.parameters(), lr=args.lr_2)
        self.criterion = torch.nn.CrossEntropyLoss().to(args.device)

        clip = torch.nn.DataParallel(clip, device_ids=[0, 1, 2, 3])
        self.model = clip.to(args.device)

        self.results_data = []
        if args.resume_file:
            assert os.path.exists(args.resume_file), 'checkpoint not found!'
            logger.info('loading model checkpoint from %s..' % args.resume_file)
            checkpoint = torch.load(args.resume_file)
            clip.load_state_dict(checkpoint['state_dict'], strict=False)
            # self.start_epoch = checkpoint['k'] + 1

    def train(self, data_dict):
        train_dataset = data_package('ContractData', data_dict['train_data'], data_dict['train_label'])
        positive_dataset = data_package('PositiveData', data_dict['train_data'], data_dict['train_label'])
        logging.info(f'Start clip training!')
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size_clip, shuffle=True, drop_last=True)
        positive_loader = DataLoader(positive_dataset, batch_size=self.args.batch_size_clip, shuffle=True,
                                     drop_last=True)
        for epoch in range(self.start_epoch, self.args.epoch_clip + self.start_epoch):
            plot_data, plot_label = self.train_epoch(epoch, train_loader, positive_loader)
        return self.gen_label(data_dict, plot_data, plot_label)

    def train_epoch(self, epoch, all_dataloader, p_dataloader):
        self.model.train()

        plot_data = []
        plot_label = []

        pbar = tqdm(zip(all_dataloader, cycle(p_dataloader)), total=len(all_dataloader))
        loss_num = 0
        for i, (data, data_p) in enumerate(pbar):
            code, label = data[0], data[1]
            code_p, label_p = data_p[0], data_p[1]

            code_token = self.tokenizer(list(code), padding=True, truncation=True, return_tensors='pt',
                                        max_length=self.text_seq_len)

            code_p_token = self.tokenizer(list(code_p), padding=True, truncation=True, return_tensors='pt',
                                          max_length=self.text_seq_len)

            ids = code_token['input_ids'].to(self.args.device)
            mask = code_token['attention_mask'].to(self.args.device)

            ids_p = code_p_token['input_ids'].to(self.args.device)
            mask_p = code_p_token['attention_mask'].to(self.args.device)

            label = label.to(self.args.device)
            label_p = label_p.to(self.args.device)

            CLS1, CLS2 = self.model(text1=ids,
                                    mask1=mask,
                                    text2=ids_p,
                                    mask2=mask_p,
                                    training_classifier=False)
            loss = self.c_loss(CLS1, CLS2, label & label_p)

            plot_data.append(CLS1)
            plot_label.append(label)

            pbar.set_description(f'epoch: {epoch}')
            # loss and step
            pbar.set_postfix(index=i, loss=loss.sum().item())
            self.optimizer.zero_grad()
            loss.sum().backward()
            self.optimizer.step()
            loss_num += loss.sum().item()
        if epoch % 5 == 0:
            self.save_picture(plot_data, plot_label, epoch)

        logger.info(f'epoch:{epoch},loss:{loss_num / len(pbar):.4f}')
        if epoch % 19 == 0:
            self.savemodel(epoch)

        return plot_data, plot_label

    def save_picture(self, plot_data, plot_label, epoch):

        if not os.path.exists(os.path.join(self.args.savepath, self.args.dataset)):
            os.mkdir(os.path.join(self.args.savepath, self.args.dataset))
        if not os.path.exists(os.path.join(self.args.savepath, self.args.dataset, "figure")):
            os.mkdir(os.path.join(self.args.savepath, self.args.dataset, "figure"))

        torch.save(torch.cat(plot_data, dim=0), f'./Results/{self.args.dataset}/{epoch}_data.pt')
        torch.save(torch.cat(plot_label), f'./Results/{self.args.dataset}/{epoch}_label.pt')

        plot_data = torch.cat(plot_data, dim=0).to('cpu').detach().numpy()
        plot_label = torch.cat(plot_label).to('cpu').detach().numpy()
        plt.scatter(plot_data[:, 0], plot_data[:, 1], c=plot_label)

        plt.savefig(
            os.path.join(self.args.savepath, self.args.dataset, "figure", f'figure_{epoch}.png'))
        logger.info(f'figure_{epoch}.png')
        plt.close()

    def savemodel(self, k):
        if not os.path.exists(os.path.join(self.args.savepath, self.args.dataset)):
            os.mkdir(os.path.join(self.args.savepath, self.args.dataset))
        torch.save({'state_dict': self.model.state_dict(),
                    'k': k,
                    'optimizer': self.optimizer.state_dict()},
                   os.path.join(self.args.savepath, self.args.dataset,
                                f'model_{k}.pth'))
        logger.info(f'save:{k}.pth')

    def decision_plot(self, ml_model, plot_data, plot_label, label_epoch):
        # 生成网格点来绘制决策边界
        h = 0.02  # 网格步长
        x_min, x_max = plot_data[:, 0].min() - 1, plot_data[:, 0].max() + 1
        y_min, y_max = plot_data[:, 1].min() - 1, plot_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = ml_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # 绘制决策边界和数据点
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        plt.scatter(plot_data[:, 0], plot_data[:, 1], c=plot_label, cmap=plt.cm.Paired)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Nonlinear SVM with RBF Kernel')
        plt.savefig(
            os.path.join(self.args.savepath, self.args.dataset, "figure", f'figure_edge_{label_epoch}.png'))
        logger.info(f'decision edge figure_{label_epoch}.png')
        plt.close()

    def gen_label(self, data_dict, plot_data, plot_label):
        unlabel_dataset = data_package('UnlabelData', data_dict['unlabel_data'], label=data_dict['true_label'])
        unlabel_loader = DataLoader(unlabel_dataset, batch_size=self.args.batch_size_clip,
                                    drop_last=True)

        logger.info('labeled_epoch: %s' % data_dict['labeled_epoch'])

        if self.args.plot_path is None:
            plot_data = torch.cat(plot_data, dim=0).to('cpu').detach().numpy()
            plot_label = torch.cat(plot_label).to('cpu').detach().numpy()
            logger.info('loading data and label!')
        else:
            data_path = rf'./Results/{self.args.dataset}/{self.args.plot_path}_data.pt'
            label_path = rf'./Results/{self.args.dataset}/{self.args.plot_path}_label.pt'
            plot_data = torch.load(data_path).detach().numpy()
            plot_label = torch.load(label_path).detach().numpy()
            logger.info(f'loading {self.args.dataset}/{self.args.plot_path}_data and {self.args.plot_path}_label!')

        scaler = StandardScaler()
        plot_data = scaler.fit_transform(plot_data)
        logger.info('NonLinear SVM train!')

        SVM_model = SVC(kernel='rbf', C=1.0, gamma=1.0)
        SVM_model.fit(plot_data, plot_label)

        self.decision_plot(ml_model=SVM_model, plot_data=plot_data, plot_label=plot_label,
                           label_epoch=data_dict['labeled_epoch'])

        pseudo_data = []
        pseudo_label = []
        unlabel_data = []

        correct_positive_label = 0
        correct_negative_label = 0

        self.model.eval()
        with torch.no_grad():
            for data, true_label in tqdm(unlabel_loader):
                token = self.tokenizer(list(data), padding=True, truncation=True, return_tensors='pt',
                                       max_length=self.text_seq_len)
                ids = token['input_ids'].to(self.args.device)
                mask = token['attention_mask'].to(self.args.device)
                outputs, _ = self.model(text1=ids, mask1=mask, text2=ids, mask2=mask)
                outputs = scaler.transform(outputs.to('cpu').detach().numpy())
                distances = SVM_model.decision_function(outputs)

                for i, distance in enumerate(distances):
                    if abs(distance) > self.args.threshold and distance > 0:
                        pseudo_data.append(data[i])
                        pseudo_label.append(1)
                        if true_label[i] == 1:
                            correct_positive_label +=1
                    elif abs(distance) > self.args.threshold and distance < 0:
                        pseudo_data.append(data[i])
                        pseudo_label.append(0)
                        if true_label[i] == 0:
                            correct_negative_label +=1
                    else:
                        unlabel_data.append(data[i])

            if sum(pseudo_label):
                logger.info(f'positive label correct rate: {correct_positive_label / sum(pseudo_label)}')
                logger.info(f'negative label correct rate: {correct_negative_label / (len(pseudo_data) - sum(pseudo_label))}')
        data_dict['unlabel_data'] = unlabel_data
        data_dict['train_data'].extend(pseudo_data)
        data_dict['train_label'].extend(pseudo_label)

        logger.info(
            f'label {len(pseudo_data)} code, {sum(pseudo_label)} positive and {len(pseudo_data) - sum(pseudo_label)} negative!')
        logger.info(f'{len(unlabel_data)} code unlabel!')
        return pseudo_data, pseudo_label, data_dict

    def train_classicication(self, data_dict):
        train_dataset = data_package('ContractData', data_dict['train_data'], data_dict['train_label'])
        test_dataset = data_package('ContractData', data_dict['test_data'], data_dict['test_label'])
        logging.info(f'Start classicition training!')
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size_clip, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size_clip, shuffle=True, drop_last=True)
        for epoch in range(self.start_epoch, self.args.epoch_cla + self.start_epoch):
            plot_data, plot_label = self.train_cla_epoch(epoch, train_loader)
            logging.info('Epoch %d finished' % epoch)
        self.eval_epoch(test_loader)
        result_df = pd.DataFrame(self.results_data, columns=['f1', 'precision', 'recall'])
        save_path = self.args.savepath + '/result_record_trymodel_' + self.args.dataset + '.csv'
        result_df.to_csv(save_path, mode='a', index=False, header=True)
        return plot_data, plot_label

    def train_cla_epoch(self, epoch, train_loader):
        self.model.train()

        loss_num = 0.0
        all_labels = []
        all_preds = []

        plot_data = []
        plot_label = []
        logger.info(f"epoch {epoch} training star!")
        pbar = tqdm(train_loader, total=len(train_loader))
        for i, (inputs, label) in enumerate(pbar):
            token = self.tokenizer(list(inputs), padding=True, truncation=True, return_tensors='pt',
                                   max_length=self.text_seq_len)
            ids = token['input_ids']
            ids, label = ids.to(self.args.device), label.to(self.args.device)
            mask = token['attention_mask']
            outputs = self.model(text1=ids,
                                 mask1=mask,
                                 training_classifier=True)
            loss = self.criterion(outputs, label)

            plot_data.append(outputs)
            plot_label.append(label)

            _, predicted = torch.max(outputs.data, dim=1)
            all_preds.extend(predicted)
            all_labels.extend(label)
            self.cla_optimizer.zero_grad()
            loss.sum().backward()
            self.cla_optimizer.step()

            loss_num += loss.sum().item()

            pbar.set_description(f'epoch: {epoch}')
            # loss and step
            pbar.set_postfix(index=i, loss=loss.sum().item())

        if epoch % 5 == 0:
            epoch_cla = f'cla_{epoch}'
            self.save_picture(plot_data, plot_label, epoch_cla)

        return plot_data, plot_label

    def eval_epoch(self, dev_loader):
        self.model.eval()

        all_labels = []
        all_preds = []
        with torch.no_grad():
            for data, label in tqdm(dev_loader):
                token = self.tokenizer(list(data), padding=True, truncation=True, return_tensors='pt',
                                       max_length=self.text_seq_len)
                ids = token['input_ids']
                ids, label = ids.to(self.args.device), label.to(self.args.device)
                outputs = self.model(text1=ids, training_classifier=True)
                _, predicted = torch.max(outputs.data, dim=1)
                all_preds.extend(predicted)
                all_labels.extend(label)

            tensor_labels, tensor_preds = torch.tensor(all_labels), torch.tensor(all_preds)
            f1, precision, recall, tp, tn, fp, fn = all_metrics(tensor_labels, tensor_preds)
            self.results_data.append([f1, precision, recall])
            logger.info(
                'Valid set -f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'
                .format(f1, precision, recall))
            logger.info('Valid set -tp: {:.4f}, tn: {:.4f}, fp: {:.4f}, fn: {:.4f}'.format(tp, tn, fp, fn))