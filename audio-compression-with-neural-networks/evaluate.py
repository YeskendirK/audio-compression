import argparse
import torch
print(torch.__version__)
import torch.utils.data
import torch.nn as nn
import torchaudio
#import torch.distributions.normal.Normal as Normal
import torch.optim as optim

import numpy as np
import re
import random
import ctypes
from pesq import *

from torch.autograd import Variable
from torchaudio import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from math import log10
from model_audio import convblock_iclr_base_single

import time
import os
import subprocess
import shutil

nBottleneck = 144

rate = 9.00


iter = 90
neck = [48, 96, 144]
#for nBottleneck in neck:
""" 
print("-----nBottleneck = " + str(nBottleneck) + "-----------")
for epoch in range(1, iter + 1):
    #for batch_idx in range(1, 174):

    if epoch%10 != 0:
        continue
        #epoch = 80
    print(epoch)
    batch_idx = 174
        #if epoch == 85:
        #    continue
    input_name =  "saved_wav/" + str(nBottleneck) + "_neck/testing/testing_" + \
                             str(epoch)+ "_" + str(batch_idx) +"_orig.wav"
    output_name = "saved_wav/" + str(nBottleneck) + "_neck/testing/testing_" + \
                      str(epoch) + "_" + str(batch_idx) + "_recon.wav"
        #input_name = "saved_wav/tested/testingg_{}_174_orig.wav".format(epoch)
        #output_name = "saved_wav/tested/testing_{}_174_recon.wav".format(epoch)

    orig, rate_orig = torchaudio.load(input_name)
    recon, rate_recon = torchaudio.load(output_name)

    pesq_val = run_pesq_waveforms(orig.numpy(), recon.numpy())


    print(pesq_val)
        #print(pesq_opus)
"""




opus_rates = [3.36, 14.81, 58.86, 118.47, 174.14]
bottle_necks = [1,12,48,96,144]
for i in range(0, len(opus_rates)):
    rate = opus_rates[i]
    nBottleneck = bottle_necks[i]
    print("-----------nBottleneck = " + str(nBottleneck) + "-----------")
    input_name = "p315.wav"
    output_name = "p315_" + str(nBottleneck) + ".wav"

    orig, rate_orig = torchaudio.load(input_name)
    recon, rate_recon = torchaudio.load(output_name)

    pesq_val = run_pesq_waveforms(orig.numpy(), recon.numpy())
    print(pesq_val)

""" 
    dest_enc = "/home/urp/py_compression/temporary/" + str(nBottleneck) + "_neck/" + str(rate) + "_" + \
                         str(epoch)+ "_" + str(batch_idx) +"_comp.opus"
    dest_wav = "/home/urp/py_compression/temporary/" + str(nBottleneck) + "_neck/" + str(rate) + "_" + \
                         str(epoch)+ "_" + str(batch_idx) +"_opus.wav"

    orig_wav = input_name#"/home/urp/py_compression/dataset_vctk/p315_001" + ".wav"

    command_encode = 'ffmpeg ' + \
                         '-i ' + orig_wav + ' -ar 48000 -ab ' + str(rate) + 'k ' + dest_enc
    command_decode = 'ffmpeg ' + \
                         '-i ' + dest_enc + ' -ar 48000 -ab ' + str(rate) + 'k ' + dest_wav
    subprocess.call(command_encode, shell=True)
    subprocess.call(command_encode, shell=True)
    subprocess.call('rm ' + dest_enc, shell=True)

    os.system(command_encode)
    os.system(command_decode)
    os.system('rm ' + dest_enc)

    orig_opus, rate_orig_opus = torchaudio.load(orig_wav)
    recon_opus, rate_recon_opus = torchaudio.load(dest_wav)

    pesq_opus = run_pesq_waveforms(orig_opus.numpy(), recon_opus.numpy())
"""