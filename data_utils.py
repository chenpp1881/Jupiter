from torch.utils.data import Dataset
from random import choice
from sklearn.model_selection import train_test_split
import logging
import json
import re

logger = logging.getLogger(__name__)


class ContractDataSet(Dataset):
    def __init__(self, data, label):
        super(ContractDataSet, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], int(self.label[idx])


class ContractPositiveDataSet(Dataset):
    def __init__(self, data, label=None):
        super(ContractPositiveDataSet, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return choice(self.data), 1

class UnlabelDataSet(Dataset):
    def __init__(self, data, label):
        super(UnlabelDataSet, self).__init__()
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], int(self.label[idx])

package = {'ContractData': ContractDataSet, 'PositiveData': ContractPositiveDataSet, 'UnlabelData':UnlabelDataSet}

def data_package(package_type, data, label):

    if package_type == 'PositiveData':
        data = positive_data(data, label)
    return package[package_type](data, label)

def positive_data(train_data, train_label):
    return [train_data[i] for i in range(len(train_label)) if train_label[i] == 1]

def load_data(args):
    if args.dataset == 'RE':
        return load_ree_data(args)

    elif args.dataset == 'TD':
        return load_time_data(args)

    elif args.dataset == 'IO':
        return load_io_data(args)
    else:
        raise ValueError('No such dataset')

def load_io_data(args):
    all_label = []
    all_data = []
    with open(r'../OurMethod/Data/IO/dataset.json', 'r', encoding='utf-8') as f:
        datas = json.load(f)

    # iter dic
    for data in datas:
        all_data.append(remove_comments(data['code']))
        # all_data.append(contract['code'])
        all_label.append(int(data['label']))

    train_data, test_data, train_label, test_label = train_test_split(all_data, all_label, test_size=0.2,
                                                                      random_state=666)

    train_data, unlabel_data, train_label, _ = split_useable_data(train_data, train_label, train_size=args.train_size)

    logger.info('IO dataset loaded successfully!')

    return {'train_data': train_data, 'train_label': train_label, 'test_data': test_data, 'test_label': test_label,
            'unlabel_data': unlabel_data, 'true_label': _}


def load_ree_data(args):
    all_label = []
    all_data = []
    with open(r'../OurMethod/Data/reentrancy/data.json', 'r', encoding='utf-8') as f:
        datas = json.load(f)

    # iter dic
    for file_id, file in datas.items():
        for contract_id, contract in file.items():
            all_data.append(remove_comments(contract['code']))
            # all_data.append(contract['code'])
            all_label.append(contract['lable'])

    train_data, test_data, train_label, test_label = train_test_split(all_data, all_label, test_size=0.2,
                                                                      random_state=666)

    train_data, unlabel_data, train_label, _ = split_useable_data(train_data, train_label, train_size=args.train_size)

    logger.info('RE dataset loaded successfully!')

    return {'train_data': train_data, 'train_label': train_label, 'test_data': test_data, 'test_label': test_label,
            'unlabel_data': unlabel_data, 'true_label': _}

def load_time_data(args):
    all_label = []
    all_data = []
    with open(r'../OurMethod/Data/timestamp/data.json', 'r', encoding='utf-8') as f:
        datas = json.load(f)

    # iter dic
    for file_id, file in datas.items():
        for contract_id, contract in file.items():
            all_data.append(remove_comments(contract['code']))
            # all_data.append(contract['code'])
            all_label.append(contract['lable'])

    train_data, test_data, train_label, test_label = train_test_split(all_data, all_label, test_size=0.2,
                                                                      random_state=666)

    train_data, unlabel_data, train_label, _ = split_useable_data(train_data, train_label, train_size=args.train_size)

    logger.info('TD dataset loaded successfully!')

    return {'train_data': train_data, 'train_label': train_label, 'test_data': test_data, 'test_label': test_label,
            'unlabel_data': unlabel_data, 'true_label': _}

def split_useable_data(data, label, train_size):
    train_data, test_data, train_label, test_label = train_test_split(data, label, train_size=train_size,
                                                                      random_state=666)

    logger.info("split dataset into %d useable data and %d unlabeled data" % (len(train_data), len(test_data)))
    return train_data, test_data, train_label, test_label


def remove_comments(code):
    # 正则表达式模式匹配Solidity注释
    pattern = r"\/\*(.|[\r\n])*?\*\/|\/\/.*"

    # 使用正则表达式替换注释为空字符串
    code_without_comments = re.sub(pattern, "", code)

    return code_without_comments
