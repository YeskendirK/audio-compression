import torch
#torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda as cuda
import math
import torch.nn.init as init

Convolution = nn.Conv2d
Tanh = nn.Tanh
ReLU = nn.ReLU
LeakyReLU = nn.LeakyReLU
upConv = nn.ConvTranspose2d

# Residual Block
class iclr_block(nn.Module):
    def __init__(self, ninput, noutput, type):
        super(iclr_block, self).__init__()
        if type == 'Tanh':
            self.act = Tanh(True)
            self.nonlinearity = 'tanh'
        elif type == 'ReLU':
            self.act = ReLU(True)
            self.nonlinearity = 'relu'
        elif type == 'Leaky':
            self.act = LeakyReLU(True)
            self.nonlinearity = 'leaky_relu'

        self.conv1 = Convolution(ninput, 128, (3,3), (1,1), (1,1))
        self.conv2 = Convolution(128, noutput, (3,3), (1,1), (1,1))
        #self.act = act
        #self.nonlinearity = nonlinearity

        self._initialize_weights()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x + residual
        return x
    def _initialize_weights(self):
        """" ""
        if type == 'Tanh':
            self.nonlinearity = 'tanh'
        elif type == 'ReLU':
            self.nonlinearity = 'relu'
        elif type == 'Leaky':
            self.nonlinearity = 'leaky_relu'
            """
        init.orthogonal(self.conv1.weight, init.calculate_gain(self.nonlinearity))
        init.orthogonal(self.conv2.weight, init.calculate_gain(self.nonlinearity))

#Residual UpBlock
class iclr_upblock(nn.Module):
    def __init__(self, ninput, noutput, type):
        super(iclr_upblock, self).__init__()
        if type == 'Tanh':
            self.act = Tanh(True)
            self.nonlinearity = 'tanh'
        elif type == 'ReLU':
            self.act = ReLU(True)
            self.nonlinearity = 'relu'
        elif type == 'Leaky':
            self.act = LeakyReLU(True)
            self.nonlinearity = 'leaky_relu'
        self.conv1 = Convolution(ninput, 128, (3, 3), (1, 1), (1, 1))
        self.conv2 = Convolution(128, noutput, (3, 3), (1, 1), (1, 1))
        #self.act = act
        #self.nonlinearity = nonlinearity
        useConv = (ninput != noutput)
        stride = 1
        if useConv:
            if stride == 1:
                self.upshortcut = upConv(ninput,noutput, (1,1), (stride, stride))
            else:
                self.upshortcut = upConv(ninput, noutput, (1, 1), (stride, stride))
        else:
            self.upshortcut = nn.AvgPool2d((1,1),(stride, stride))

        self._initialize_weights()

    def  forward(self,x):
        residual = self.upshortcut(x)

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x + residual
        return x
    def _initialize_weights(self):
        #init.orthogonal(self.upshortcut.weight, init.calculate_gain(self.nonlinearity))
        init.orthogonal(self.conv1.weight, init.calculate_gain(self.nonlinearity))
        init.orthogonal(self.conv2.weight, init.calculate_gain(self.nonlinearity))



# Gaussian Scale Mixture
"""
class gsm_model(nn.Module):
    def __init__(self,  nwidth, nChannel, nScale):
        super(gsm_model, self).__init__()
        # Randomly initialize the parameters
        self.var = Variable(torch.randn(nChannel, nScale), requires_grad = True)
        self.phi = F.softmax(Variable(torch.randn(nChannel, nScale), requires_grad=True), 1)
        self.nScale = nScale
        self.nChannel = nChannel
        self.nwidth = nwidth
        
    def forward(self, x):
        """
"""
        covar_inv = 1. / self.var

        det = (2*np.pi*self.var).prod(dim = 1)
        coeff = 1. / det.sqrt()

        x = x.unsqueeze(0).repeat(self.nScale,1,1,1,1)

        exponent = (x ** 2).mm(covar_inv.unsqueeze(2))
        exponent = -0.5 * exponent

        P = coeff.view(1,1) * exponent.exp()
        P = P.squeeze(2)
        sum_over_s = torch.sum(self.phi.unsqueeze(2)*P, dim=0)
        sum_over_k = torch.sum(torch.log(sum_over_s))
        """
"""
        covar_inv = 1./self.var
        det = (2*np.pi*self.var)
        coeff = 1./det.sqrt()
        x = x.unsqueeze(0).repeat(self.nScale, 1,1,1,1)
        covar_inv = covar_inv.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        exponent = (x ** 2).mm(covar_inv)
        coeff = coeff.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        P = coeff* exponent.exp()
        self.phi = self.phi.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        sum_over_s = torch.sum(self.phi * P, dim=0)
        sum_over_k = torch.sum(torch.sum(torch.log(sum_over_s),dim = 1), dim = 2)
        sum_over_k = torch.sum(sum_over_k, dim = 3)
        sum_over_k = torch.sum(sum_over_k)
        return sum_over_k
"""

""" ""
class encoder(nn.Module):
    def __init__(self, nBottleneck, nChannel, type):
        super(encoder. self).__init__()
        self.en_conv1 = Convolution(3,64,(5,5),(2,2),(2,2))
        self.leaky_relu = LeakyReLU(True)
        self.iclr_block = iclr_block(128,128,type)
        self.en_conv2 = Convolution(128, nBottleneck, (5, 5), (2, 2), (2, 2))

    def forward(self, x):
        x = self.en_conv1(x)
        x = self.leaky_relu(x)
        x = self.en_conv1(x)
        x = self.leaky_relu(x)
        x = self.iclr_block.forward(x)
        x = self.iclr_block.forward(x)
        x = self.iclr_block.forward(x)
        x = self.en_conv2(x)
        x = self.leaky_relu(x)
        return x

class decoder(nn.Module):
    def __init__(self, nBottleneck, nChannel, type):
        super(decoder, self).__init__()
        self.upConv1 = upConv(nBottleneck,128,(2,2),(2,2),(0,0))
        self.upConv2 = upConv(128,64,2,2,2,2,0,0)
        self.upConv3 = upConv(64,3,2,2,2,2,0,0)
        self.iclr_upblock = iclr_upblock(128,128,type)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.de_conv1 = Convolution(nBottleneck,512,3,3,1,1,1,1)
        self.de_conv2 = Convolution(128,256,3,3,1,1,1,1)
        self.de_conv3 = Convolution(64, 12, (3, 3), (1, 1), (1, 1))
        self.leaky_relu = LeakyReLU(True)


    def forward(self, x):
        if opt.pixel_shuffle == True:
            x = self.conv1(x)
            x = self.pixel_shuffle(x)
            x = self.iclr_upblock.forward(x)
            x = self.iclr_upblock.forward(x)
            x = self.iclr_upblock.forward(x)
            x = self.conv2(x)
            x = self.pixel_shuffle(x)
            x = self.leaky_relu(x)
            x = self.conv3(x)
            x = self.pixel_shuffle(x)
        else:
            x = self.upConv1(x)
            x = self.iclr_upblock.forward(x)
            x = self.iclr_upblock.forward(x)
            x = self.iclr_upblock.forward(x)
            x = self.upConv2(x)
            x = self.leaky_relu(x)
            x = self.upConv3(x)

        return x
"""

class convblock_iclr_base_single(nn.Module):
    def __init__(self, nBottleneck, nChannel, type):
        super(convblock_iclr_base_single, self).__init__()
        """ ""
        encoder = nn.Sequential()
        encoder.add_module('conv1', Convolution(3,64,(5,5),(2,2),(2,2))).add_module(LeakyReLU(True))
        encoder.add_module('conv2', Convolution(3, 64, (5, 5), (2, 2), (2, 2))).add_module(LeakyReLU(True))
        encoder.add_module('iclr_block1', iclr_block(128,128,type))
        encoder.add_module('iclr_block2', iclr_block(128, 128, type))
        encoder.add_module('iclr_block3', iclr_block(128, 128, type))
        encoder.add_module('conv3', Convolution(128, nBottleneck, (5, 5), (2, 2), (2, 2))).add_module(LeakyReLU(True))
        ##################################################
        ##encoder.add_module(nn.StochasticRound())
        decoder = nn.Sequential()
        decoder.add_module('deconvolution',upConv(nBottleneck,128,(2,2),(2,2),(0,0)))
        decoder.add_module('iclr_upblock1',iclr_upblock(128,128,type))
        decoder.add_module('iclr_upblock2', iclr_upblock(128, 128, type))
        decoder.add_module('iclr_upblock3', iclr_upblock(128, 128, type))
        decoder.add_module(Convolution(64,12,(3,3),(1,1),(1,1)))
        decoder.add_module(nn.PixelShuffle(2))
        
        self.encoder = encoder
        self.decoder = decoder
        """
        #for ENCODER
        self.en_conv1 = Convolution(3, 64, (5, 5), (2, 2), (2, 2))
        self.en_conv2 = Convolution(64, 128, (5, 5), (2, 2), (2, 2))
        self.leaky_relu = LeakyReLU(True)
        self.iclr_block = iclr_block(128, 128, type)
        self.en_conv3 = Convolution(128, nBottleneck, (5, 5), (2, 2), (2, 2))

        #for DECODER
        self.upConv1 = upConv(nBottleneck, 128, (2, 2), (2, 2), (0, 0))
        self.upConv2 = upConv(128, 64, (2, 2), (2, 2), (0, 0))
        self.upConv3 = upConv(64, 3, (2, 2), (2, 2), (0, 0))
        self.iclr_upblock = iclr_upblock(128, 128, type)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.de_conv1 = Convolution(nBottleneck, 512, (3, 3), (1, 1), (1, 1))
        self.de_conv2 = Convolution(128, 256, (3, 3), (1, 1), (1, 1))
        self.de_conv3 = Convolution(64, 12, (3, 3), (1, 1), (1, 1))
        self.leaky_relu = LeakyReLU(True)

        #for GSM_MODEL
        #self.gsm_model = gsm_model()
        self.nBottleneck = nBottleneck


    def encoder(self,x):
        x = self.en_conv1(x)
        x = self.leaky_relu(x)
        x = self.en_conv2(x)
        x = self.leaky_relu(x)
        x = self.iclr_block.forward(x)
        x = self.iclr_block.forward(x)
        x = self.iclr_block.forward(x)
        x = self.en_conv3(x)
        x = self.leaky_relu(x)
        return x

    def decoder(self, x):
        if True:

            x = self.de_conv1(x)
            x = self.pixel_shuffle(x)
            x = self.iclr_upblock.forward(x)
            x = self.iclr_upblock.forward(x)
            x = self.iclr_upblock.forward(x)
            x = self.de_conv2(x)
            x = self.pixel_shuffle(x)
            x = self.leaky_relu(x)
            x = self.de_conv3(x)
            x = self.pixel_shuffle(x)
        """" 
        else:
            x = self.upConv1(x)
            x = self.iclr_upblock.forward(x)
            x = self.iclr_upblock.forward(x)
            x = self.iclr_upblock.forward(x)
            x = self.upConv2(x)
            x = self.leaky_relu(x)
            x = self.upConv3(x)
        """
        return x

    def forward(self,x):
        e = self.encoder(x)
        d = self.decoder(e)
        #gsm = gsm_model(16, self.nBottleneck, 6).forward(e)
        #y = [e,d, gsm]
        y = [e, d]
        return y












