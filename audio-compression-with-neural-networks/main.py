import argparse
import torch
print(torch.__version__)
#torch.backends.cudnn.enabled = False
import torch.utils.data
import torch.nn as nn
#import torch.distributions.normal.Normal as Normal
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from math import log10
from model import convblock_iclr_base_single
#paths.dofile('eval/utils/util.py')
import os, sys
sys.path.append('/mnt/home/20140941/py_compression/eval/utils')
from util import *
from sumCriterion import *
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

opt  = argparse.ArgumentParser(description = "Compression with baseline")
opt.add_argument('--dataset', default='normal',
                 help='dataset name, |NC_color|NC_spec|NC_normal|flickr|')
opt.add_argument('--nThreads', default = 1, help='n threads')
opt.add_argument('--loadsize', default= 128, help = 'load size')
opt.add_argument('--ntrain', default = 1, help = 'ntrain')
opt.add_argument('--model-folder', default ='models/', help = 'model folder')
opt.add_argument('--png_folder', default = 'recon/', help = 'recon folder')
opt.add_argument('--gsm', default ='True', help = 'gsm model')
opt.add_argument('--mtype', default = 'base_single',
                 help = 'model type |aux|base|wave|')
opt.add_argument('--beta', default = 0.03, help = 'beta (ration between loss)')
opt.add_argument('--lr', default = 0.0001, help = 'learning rate')
opt.add_argument('--batchSize', default = 32, help = 'batch Size')
opt.add_argument('--beta1', default = 0.9, help = 'adam param')
opt.add_argument('--nC', default = 128, help = 'channel size')
opt.add_argument('--niter', default = 200, help = 'n epoch')
opt.add_argument('--pixel_shuffle', default = True, help = 'pixel shuffle')
opt.add_argument('--nBottleneck', default = 48,
                 help = 'number of bottleneck channel')
opt.add_argument('--gpu', default = 1, help = 'n gpu')
opt = opt.parse_args()


if (opt.gpu > 0) and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(1)
if opt.gpu > 0:
    torch.cuda.manual_seed(1)

print('===> Loading datasets')
train_set = datasets.CIFAR10('../data',train = True, download = True,
                           transform=transforms.ToTensor())
training_data_loader = DataLoader(dataset = train_set,
            num_workers=opt.nThreads, batch_size=opt.batchSize,shuffle=True)

test_set = datasets.CIFAR10('../data',train = False, download = True,
                           transform=transforms.ToTensor())
testing_data_loader = DataLoader(dataset = test_set,
            num_workers=opt.nThreads, batch_size=opt.batchSize,shuffle=True)

print('===> Building model')
model = convblock_iclr_base_single(opt.nBottleneck, opt.nC, 'Leaky')
criterion_mse = nn.MSELoss()
#criterion_p = PSNRCriterion()
#criterion_s = SSIMCriterion()
criterion_sum = sumCriterion()

if opt.gpu > 0:
    model = model.cuda()
    criterion_mse = criterion_mse.cuda()
    #criterion_p = criterion_p.cuda()
    #criterion_s = criterion_s.cuda()
    criterion_sum = criterion_sum.cuda()

optimizer = optim.Adam(list(model.parameters()), lr=opt.lr)
"""" ""
def gsm_model(encoder, nwidth, nChannel, nScale):
    #nwidth = 16, nChannel = nBottleneck, nScale = 6
    sigma = Variable(torch.randn(nChannel, nScale))
    phi = Variable(torch.randn(nChannel, nScale))
    normal_distrib = Normal(encoder, sigma)
"""


def loss_function(recon_x, x):
    err = criterion_mse(recon_x, x)
    return err

def average_criterion(x, y):
    err = (1-opt.beta) * x + opt.beta * y
    return err

def train(epoch):
    model.train()
    train_loss = 0
    avg_psnr = 0
    for batch_idx, batch in enumerate(training_data_loader):
        input = Variable(batch[0])
        if opt.gpu > 0:
            input = input.cuda()
        optimizer.zero_grad()
        output = denormalize( model(normalize(input, mean, std))[1], mean, std)

        loss = loss_function(output, input)
        mse = criterion_mse(output, input)
        psnr = 10 * log10(1/mse.data[0])
        #gsm_result = model(normalize(input, mean, std))[2]
        #if opt.gsm:
        #    loss = average_criterion(mse, criterion_sum(gsm_result))
        #else:
        #    loss = loss_function(output, input)

        #gsm = model(normalize(input, mean, std))[2]
        #print(type(gsm_result))


        loss.backward()
        train_loss += loss.data[0]
        avg_psnr += psnr
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "===> Epoch[{}]({}/{}): Loss: {:.4f} ".format(epoch,
                batch_idx, len(training_data_loader), loss.data[0]) )

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch,
                            train_loss / len(training_data_loader)))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr /
                                             len(training_data_loader)))


def test(epoch):
    model.eval()
    test_loss = 0
    avg_psnr = 0
    for i, batch in enumerate(testing_data_loader):
        input = Variable(batch[0])#, volatile = True)
        if opt.gpu > 0:
            input = input.cuda()
        output = denormalize(model(normalize(input, mean, std))[1], mean, std)
        test_loss += loss_function(output, input).data[0]
        mse = criterion_mse(output, input)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
        """ ""
        if i == 0:
            n = min(input.size(0), 8)
            comparison = torch.cat([input[:n],
                                    output.view(opt.batchSize,3,32,32)])
            save_image(comparison.data.cpu(),
                       'results/reconstruction_' + str(epoch) + '.png', nrow=n)
        """
    test_loss /= len(testing_data_loader.dataset)
    print('===> Test set loss: {:.4f}'.format(test_loss))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr /
                                             len(testing_data_loader)))

def checkpoint(epoch):
    model_out_path = "test_results/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


print('===> Running')
opt.niter = 1
for epoch in range(1, opt.niter + 1):
    train(epoch)
    test(epoch)
    checkpoint(epoch)








# denormaliza - normalize
""" ""
def denormalize(img, mean, std):
    input = img.clone()
    if input.size().size() == 3:
        for i in range(input.size(1)):
            input[i].mul_(std[i])
            input[i].add_(mean[i])
    else:
        for i in range(input.size(2)):
            input[:,i].mul_(std[i])
            input[:,i].add_(mean[i])

    return input

def normalize(img, mean, std):
    input = img.clone()
    if input.size().size() == 3:
        for i in range(size(1)):
            input[i].add_(-mean[i])
            input[i].div_(std[i])
    else:
        for i in range(size(2)):
            input[:,i].add_(-mean[i])
            input[:,i].div_(std[i])

    return input
""""s"
