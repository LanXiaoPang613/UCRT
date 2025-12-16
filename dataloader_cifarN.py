from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os
from autoaugment import CIFAR10Policy, ImageNetPolicy
from torchnet.meter import AUCMeter

# import dill


## If you want to use the weights and biases
# import wandb
# wandb.init(project="noisy-label-project", entity="....")


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset):
    def __init__(self, dataset, noise_mode, root_dir, transform, mode, noise_path='',
                 is_human=True, pred=[], probability=[], log=''):

        self.noise_path = noise_path
        self.is_human=is_human
        self.transform = transform
        self.mode = mode
        root_dir_save = root_dir
        self.noise_mode = noise_mode
        self.save_file = None

        if dataset == 'cifar10':
            num_class = 10
        else:
            num_class = 100

        num_sample = 50000
        self.class_ind = {}

        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/cifar-10-batches-py/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                # root_dir = './data/cifar100/'
                test_dic = unpickle('%s/cifar-100-python/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']

        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/cifar-10-batches-py/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/cifar-100-python/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            self.clean_labels = np.array(train_label)
            self.train_label = self.clean_labels
            noise_label = []
            if noise_mode != 'clean_label':
                if os.path.exists(noise_path):
                    all_label = torch.load(self.noise_path)
                    if isinstance(all_label, dict):
                        if "clean_label" in all_label.keys():
                            clean_label = torch.tensor(all_label['clean_label'])
                            assert torch.sum(torch.tensor(train_label) - clean_label) == 0
                            print(f'Loaded {self.noise_mode} from {self.noise_path}.')
                            print(f'The overall noise rate is {1 - np.mean(clean_label.numpy() == all_label[self.noise_mode])}')
                        noise_label = all_label[self.noise_mode].reshape(-1)
                    else:
                        raise Exception('Input Error')

                    # if not is_human:
                    #     noise_label = noise_label.tolist()
                    #     synthetic_path = 'synthetic_path' + '_' + str(dataset) + '_' + str(noise_mode) + '.npz'
                    #     if os.path.exists(synthetic_path):
                    #         T = np.load(synthetic_path)
                    #         print(f'Noise transition matrix is \n{float(T)}')
                    #     else:
                    #         T = np.zeros((num_class, num_class))
                    #         for i in range(len(noise_label)):
                    #             T[train_label[i]][noise_label[i]] += 1
                    #         T = T/np.sum(T, axis=1)
                    #         print(f'Noise transition matrix is \n{float(T)}')
                    #         # 用实际的T生成合成噪声
                    #         noise_label = multiclass_noisify(y=np.array(train_label), P=T, random_state=0)
                    #         train_noisy_labels = noise_label.tolist()
                    #         T = np.zeros((num_class, num_class))
                    #         for i in range(len(train_noisy_labels)):
                    #             T[train_label[i]][train_noisy_labels[i]] += 1
                    #         T = T/np.sum(T,axis=1)
                    #         np.save(synthetic_path, T)
                    #         print(f'New synthetic noise transition matrix is \n{float(T)}')

                else:  ## Inject Noise
                    raise Exception('Path Error')
            else:
                noise_label = train_label

            noise_label = np.array(noise_label).astype(np.int64)

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = np.array(noise_label).astype(np.int64)
            else:

                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]

                    self.probability = [probability[i] for i in pred_idx]
                    clean = (np.array(noise_label) == np.array(train_label))
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability, clean)
                    auc, _, _ = auc_meter.value()
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n' % (pred.sum(), auc))
                    log.flush()

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            image = Image.fromarray(img)
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)

            return img1, img2, img3, img4, target, prob

        elif self.mode == 'unlabeled':
            img = self.train_data[index]
            image = Image.fromarray(img)
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)
            return img1, img2, img3, img4

        elif self.mode == 'all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index

        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataloader():
    def __init__(self, dataset, noise_mode, batch_size, num_workers, root_dir, log):
        self.dataset = dataset
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log

        if self.dataset == 'cifar10':
            self.noise_path = './data/CIFAR-10_human.pt'
        elif self.dataset == 'cifar100':
            self.noise_path = './data/CIFAR-100_human.pt'
        else:
            raise NameError(f'Undefined dataset {self.dataset}')

        if self.dataset == 'cifar10':
            transform_weak_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            transform_strong_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            self.transforms = {
                "warmup": transform_weak_10,
                "unlabeled": [
                    transform_weak_10,
                    transform_weak_10,
                    transform_strong_10,
                    transform_strong_10
                ],
                "labeled": [
                    transform_weak_10,
                    transform_weak_10,
                    transform_strong_10,
                    transform_strong_10
                ],
            }

            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        elif self.dataset == 'cifar100':
            transform_weak_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

            transform_strong_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

            self.transforms = {
                "warmup": transform_weak_100,
                "unlabeled": [
                    transform_weak_100,
                    transform_weak_100,
                    transform_strong_100,
                    transform_strong_100
                ],
                "labeled": [
                    transform_weak_100,
                    transform_weak_100,
                    transform_strong_100,
                    transform_strong_100
                ],
            }
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

    def run(self, mode, pred=[], prob=[], is_human=True):
        if mode == 'warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode,
                                        root_dir=self.root_dir, transform=self.transforms["warmup"],
                                        mode="all", noise_path=self.noise_path, is_human=is_human)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode,
                                            root_dir=self.root_dir, transform=self.transforms["labeled"],
                                            mode="labeled", noise_path=self.noise_path, is_human=is_human, pred=pred, probability=prob,
                                            log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, drop_last=True)

            unlabeled_dataset = cifar_dataset(dataset=self.dataset,
                                              noise_mode=self.noise_mode, root_dir=self.root_dir,
                                              transform=self.transforms["unlabeled"], mode="unlabeled",
                                              noise_path=self.noise_path, is_human=is_human, pred=pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, drop_last=True)
            unlabeled_trainloader_map = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader, unlabeled_trainloader_map

        elif mode == 'test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size*2,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='all',
                                         noise_path=self.noise_path, is_human=is_human)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=100,
                shuffle=False,
                num_workers=self.num_workers, drop_last=True)
            return eval_loader
