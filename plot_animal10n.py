from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os
import matplotlib
from autoaugment import CIFAR10Policy

def unpickle(file):
    fo = open(file, 'rb').read()
    size = 64 * 64 * 3 + 1
    for i in range(50000):
        arr = np.fromstring(fo[i * size:(i + 1) * size], dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3, 64, 64)).transpose((1, 2, 0))
    return img, lab

class animal_dataset(Dataset):
    def __init__(self, root, mode, num_class=10):

        self.root = root
        self.mode = mode
        self.noise_label = []
        self.train_dir = root + '/training/'
        self.test_dir = root + '/testing/'
        train_imgs = os.listdir(self.train_dir)
        test_imgs = os.listdir(self.test_dir)
        self.test_data = []
        self.test_labels = []
        noise_file1 = './training_batch.json'
        noise_file2 = './testing_batch.json'
        if mode == 'test':
            if os.path.exists(noise_file2):
                dict = json.load(open(noise_file2, "r"))
                self.test_labels = dict['data']
                self.test_data = dict['label']
            else:
                for img in test_imgs:
                    self.test_data.append(self.test_dir+img)
                    self.test_labels.append(int(img[0]))
                dicts = {}
                dicts['data'] = self.test_data
                dicts['label'] = self.test_labels
                # json.dump(dicts, open(noise_file2, "w"))
        else:
            if 0:
                dict = json.load(open(noise_file1, "r"))
                train_data = dict['data']
                train_labels = dict['label']
                for ip in train_data:
                    self.noise_label.append(train_labels[ip])
            else:
                train_data = []
                train_labels = {}
                for img in train_imgs:
                    img_path = self.train_dir+img
                    train_data.append(img_path)
                    train_labels[img_path] = (int(img[0]))
                    self.noise_label.append((int(img[0])))
                self.noise_label = np.array(self.noise_label).astype(np.int64)
                dicts = {}
                dicts['data'] = train_data
                dicts['label'] = train_labels
                random.shuffle(train_data)
                # json.dump(dicts, open(noise_file1, "w"))
            if self.mode == "train":
                self.train_data = train_data
                self.train_labels = train_labels

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.train_data[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            return image, target,index
        elif self.mode == 'test':
            img_path = self.test_data[index]
            target = self.test_labels[index]
            image = Image.open(img_path).convert('RGB')
            return image, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_data)
        else:
            return len(self.train_data)

import torchvision
from PIL import Image
import random
import matplotlib.pyplot as plt
train_dataset = animal_dataset\
    ('C:/Users/Administrator/Desktop/DatasetAll/Animal-10N', 'train')

classes = ['cat',
  'lynx',
  'wolf',
  'coyote',
  'cheetah',
  'jaguer',
  'chimpanzee',
  'orangutan',
  'hamster',
  'guinea pig',
           ]


def imshow(img):
    # img = img / 2 + 0.5
    npimg = img.numpy()
    '''
    img 格式： channels,imageSize,imageSize
    imshow需要格式：imageSize,imageSize,channels
    np.transpose 转换数组
    '''
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.imshow(npimg.astype(np.uint8))
    X_label = []
    for i in range(num_class):
        X_label.append(32.+64.*i)
    plt.yticks(X_label, labels=classes)
    plt.xticks([])
    plt.savefig('./Fig-3a' + '.tif', dpi=600, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
    plt.savefig('./Fig-3a' + '.svg', dpi=1000, format="svg")
    plt.show()


# dataIter = iter(train_loader)
# images, labels = dataIter.next()
num_class = 10
idx_list = []
for i in range(num_class):
    count = 0
    for j in train_dataset.train_data:
        tp_label = train_dataset.train_labels[j]
        if tp_label == i and count < num_class:
            idx_list.append(j)
            count += 1
        elif count >= num_class:
            break


transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        )
images = []
root = 'C:/Users/Administrator/Desktop/DatasetAll/WebVision1.0/'
for img_path in range(len(idx_list)):
     ttt = Image.open(idx_list[img_path]).convert("RGB")
     img = transform(ttt)
     images.append(img)

# 拼接图像：make_grid
ddd = torchvision.utils.make_grid(images, nrow=num_class)
imshow(ddd)