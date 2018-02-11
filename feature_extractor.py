from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--dataroot', required=True)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--netD', default='')
parser.add_argument('--outf', default='.')
parser.add_argument('--manualSeed', type=int)

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.set_device(1L)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

imageSize = 64
ndf = 64
nc = 3

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        self.ngpu = 1
        self.main = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


netD = _netD()
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

input = torch.FloatTensor(1, 3, imageSize, imageSize)
label = torch.FloatTensor(1)

netD.cuda()
input, label = input.cuda(), label.cuda()


dataset_root = opt.dataroot
transform = transforms.Compose(([transforms.Resize(imageSize), transforms.CenterCrop(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

import json
json_data=open(dataset_root + "/../meta/train.json").read()
train_data = json.loads(json_data)
suffix = ".jpg"
from PIL import Image

class _netDFeatures(nn.Module):
    def __init__(self):
        super(_netDFeatures, self).__init__()
        # Stop before last layer
        self.main = nn.Sequential(*list(list(netD.children())[0].children())[:-2])

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

netDFeatures = _netDFeatures()

print(netD)
print(netDFeatures)

i = 0
j = 0

num_classes = 101
data_per_class = 750
num_features = 8192

X = np.empty([num_classes * data_per_class, num_features])
y = np.empty([num_classes * data_per_class, 1]).ravel()
for key in train_data.keys():
    for example in train_data[key]:
        with open(os.path.join(dataset_root,example+suffix)) as f:
            img = Image.open(f).convert('RGB')
        imgv = Variable(transform(img).unsqueeze(0)).type(torch.FloatTensor).cuda()
        out = netDFeatures.forward(imgv)

        numpy_out = out.data.cpu().numpy()
        X[i,:] = out.data.cpu().numpy()
        y[i] = j + 1
        i = i + 1
        sys.stdout.write("\r%d" % i)
        sys.stdout.flush()
    j = j + 1

print("")

np.save("data.dat", X)
np.save("labels.dat", y)

json_data=open(dataset_root + "/../meta/test.json").read()
test_data = json.loads(json_data)
suffix = ".jpg"

i = 0
j = 0

num_classes = 101
data_per_class = 250
num_features = 8192

test_per_class = 250
X_test = np.empty([num_classes * test_per_class, num_features])
y_test = np.empty([num_classes * test_per_class, 1]).ravel()
for key in test_data.keys():
    for example in test_data[key]:
        with open(os.path.join(dataset_root,example+suffix)) as f:
            img = Image.open(f).convert('RGB')
        imgv = Variable(transform(img).unsqueeze(0)).type(torch.FloatTensor).cuda()
        out = netDFeatures.forward(imgv)

        numpy_out = out.data.cpu().numpy()
        X_test[i,:] = out.data.cpu().numpy()
        y_test[i] = j + 1
        i = i + 1
        sys.stdout.write("\r%d" % i)
        sys.stdout.flush()
    j = j + 1

np.save("test.dat", X_test)
np.save("labels_test.dat", y_test)

print("")

print("Done")
