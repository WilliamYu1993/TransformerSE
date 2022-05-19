import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from preprocess import SignalToFrames, ToTensor
import torchaudio
torchaudio.set_audio_backend("sox_io")
import os
import sys
import random
import pdb
import numpy as np
import json

class VoiceBankDemandDataset(Dataset):
    def __init__(self, data_dir, train=True, hop_length=256):
        self.data_dir = data_dir
        self.train = train
        self.tier = 'trainset_28spk' if train else 'testset'
        self.hop_length = hop_length

        self.clean_root = os.path.join(
                self.data_dir, f'clean_{self.tier}_wav_16k')
        self.noisy_root = os.path.join(
                self.data_dir, f'noisy_{self.tier}_wav_16k')

        self.clean_path = self.get_path(self.clean_root)
        if self.train:
            random.shuffle(self.clean_path)
        
    def get_path(self, root):
        paths = []
        for r, dirs, files in os.walk(root):
            for f in files:
                if f.endswith('.pt') or f.endswith('.wav'):
                    paths.append(os.path.join(r, f))
        return paths

    def padding(self, x):
        len_x = x.size(-1)
        pad_len = self.hop_length - len_x % self.hop_length
        x = F.pad(x, (0, pad_len))
        return x

    def normalize(self, x):
        return 2 * (x - x.min()) / (x.max() - x.min()) - 1

    def __len__(self):
        return len(self.clean_path)

    def __getitem__(self, idx):

        if self.train:
            cpath=self.clean_path[idx]
            npath=cpath.replace("clean_trainset", "noisy_trainset")
            clean = torchaudio.load(cpath)[0]
            noisy = torchaudio.load(npath)[0]
            length = clean.size(-1)
            frames = int(length/256)
            start = torch.randint(0, frames - 63 - 1, (1, ))
            end = start + 63
            clean = clean[:,start*256:end*256]
            noisy = noisy[:,start*256:end*256]
            return noisy, clean, length
        else:
            cpath=self.clean_path[idx]
            npath=cpath.replace("clean_testset", "noisy_testset")
            clean = torchaudio.load(cpath)[0]
            noisy = torchaudio.load(npath)[0]
            length = clean.size(-1)
            name = cpath.split('/')[-1]
            clean = self.padding(clean)
            noisy = self.padding(noisy)

        return noisy, clean, name, length
