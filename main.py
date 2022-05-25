import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchcontrib.optim import SWA
from torch.nn.utils import rnn
from torch.utils.data import DataLoader
from conv_stft_ecocatzh import STFT
from torch.utils.tensorboard import SummaryWriter
from transformerencoder import transformerencoder_wav

import os
import sys
import pdb
import json
import argparse
import tensorflow as tf
from argparse import Namespace
import numpy as np
from pesq import pesq
from tqdm import tqdm
from colorama import Fore
from collections import OrderedDict
from multiprocessing import Pool

from dataset import VoiceBankDemandDataset

def rnn_collate(batch):

    n = rnn.pad_sequence([b[0] for b in batch]).transpose(0, 1)
    c = rnn.pad_sequence([b[1] for b in batch]).transpose(0, 1)
    l = torch.LongTensor([b[2] for b in batch])

    return n, c, l

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cal_pesq(x, y, l):
    try:
        score = pesq(16000, y, x, 'wb')
    except:
        score = 0.
    #print(score)
    del x, y
    return score

def evaluate(x, y, lens, fn):
    y = list([y])#(y.cpu().detach().numpy())
    x = list([x])#(x.cpu().detach().numpy())
    lens = list(lens)
    pool = Pool(processes=args.num_workers)
    try:
        ret = pool.starmap(
            fn,
            iter([(deg, ref, l) for deg, ref, l in zip(x, y, lens)])
        )
        pool.close()
        del x, y
        return torch.FloatTensor(ret).mean()

    except KeyboardInterrupt:
        pool.terminate()
        pool.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # system setting
    parser.add_argument('--exp_dir', default=os.getcwd(), type=str)
    parser.add_argument('--exp_name', default='logs', type=str)
    parser.add_argument('--data_dir', default='/mnt/md2/user_chengyu/Corpus/TIMIT_SE/', type=str)
    parser.add_argument('--num_workers', default=3, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--add_graph', action='store_true')
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--seed', default=999, type=int)

    # training specifics
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--grad_accumulate_batches', default=1, type=int)
    parser.add_argument('--log_grad_norm', action='store_true')
    parser.add_argument('--resume_dir', default='', type=str)
    parser.add_argument('--use_swa', action='store_true')
    parser.add_argument('--use_logstftmagloss', action='store_true')
    parser.add_argument('--lr_decay', default=1.0, type=float)

    # stft/istft settings
    parser.add_argument('--n_fft', default=512, type=int)
    parser.add_argument('--hop_length', default=256, type=int)

    args = parser.parse_args()

    # add hyperparameters
    ckpt_path = os.path.join(args.exp_dir, args.exp_name, 'ckpt')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
        os.makedirs(ckpt_path.replace('ckpt', 'logs'))
        with open(os.path.join(ckpt_path, 'hparams.json'), 'w') as f:
            json.dump(vars(args), f)
    else:
        print(f'Experiment {args.exp_name} already exists.')
        sys.exit()

    writer = SummaryWriter(os.path.join(args.exp_dir, args.exp_name, 'logs'))
    writer.add_hparams(vars(args), dict())

    # seed
    if args.seed:
        fix_seed(args.seed)

    # device
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    if device == 'cuda':
        print(f'DEVICE: [{torch.cuda.current_device()}] {torch.cuda.get_device_name()}')
    else:
        print(f'DEVICE: CPU')

    
    train_dataloader = DataLoader(
        VoiceBankDemandDataset(data_dir=args.data_dir, train=True, hop_length = args.hop_length),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    test_dataloader = DataLoader(
        VoiceBankDemandDataset(data_dir="/mnt/2TB/user_chengyu/Corpus/noisy-vctk-16k", train=False, hop_length = args.hop_length),
        # batch_size=args.batch_size,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    stft = STFT(fft_len=args.nfft, win_hop=args.hop_length, win_len=args.nfft, win_type="hamming",).to(device)
    se_net = transformerencoder_wav()
    se_net.to(device)

    optimizer = optim.Adam(se_net.parameters(), lr=64**(-0.5), weight_decay=1e-7)    
    scheduler = None

    # add graph to tensorboard
    if args.add_graph:
        dummy = torch.randn(16, 1, args.hop_length * 16).to(device)
        writer.add_graph(se_net, dummy)

    start_epoch = 0
    total_loss = 0.0
    best_pesq = 0.0
    for epoch in range(start_epoch, start_epoch + args.num_epochs, 1):
        # ------------- training -------------
        se_net.train()
        pbar = tqdm(train_dataloader, bar_format='{l_bar}%s{bar}%s{r_bar}'%(Fore.BLUE, Fore.RESET), ascii=True)
        pbar.set_description(f'Epoch {epoch + 1}')
        total_loss = 0.0
        if args.log_grad_norm:
            total_norm = 0.0
        se_net.zero_grad()
        for i, (noisy, clean,  _) in enumerate(pbar):
            optimizer.zero_grad()

            ewav = se_net(noisy.cuda())
            emag, ephase = stft.transform(ewav, return_type='magphase')
            cmag, cphase = stft.transform(clean.cuda(), return_type='magphase')
            
            loss = torch.nn.L1Loss()(torch.log1p(emag), torch.log1p(cmag))# + loss_wav

            loss.backward()
            optimizer.step(epoch)

            # log metrics
            pbar_dict = OrderedDict({
                'loss': loss.item(),
            })
            pbar.set_postfix(pbar_dict)
            
            total_loss += loss.item()
            #pdb.set_trace()
            if (i + 1) % args.log_interval == 0:
                step = epoch * len(train_dataloader) + i
                writer.add_scalar('Loss/train', total_loss / args.log_interval, step+1)
                total_loss = 0.0
                
                # log gradient norm
                if args.log_grad_norm:
                    for p in se_net.parameters():
                        if p.requires_grad:
                            try:
                                norm = p.grad.data.norm(2)
                                total_norm += norm.item() ** 2
                            except:
                                pass

                    norm = total_norm ** 0.5
                    writer.add_scalar('Gradient 2-Norm/train', norm, step+1)
                    total_norm = 0.0

        # ------------- validation -------------
        pbar = tqdm(test_dataloader, bar_format='{l_bar}%s{bar}%s{r_bar}'%(Fore.LIGHTMAGENTA_EX, Fore.RESET), ascii=True)
        pbar.set_description('Validation')
        total_loss, total_pesq = 0.0, 0.0
        num_test_data = len(test_dataloader)
        with torch.no_grad():
            se_net.eval()
            for i, (noisy, clean, _, l) in enumerate(pbar):

                ewav = se_net(noisy.cuda())

                emag, ephase = stft.transform(ewav, return_type='magphase')
                cmag, cphase = stft.transform(clean.cuda(), return_type='magphase')
            
                loss = torch.nn.L1Loss()(torch.log1p(emag), torch.log1p(cmag))# + loss_wav

                pesq_score = evaluate(ewav.cpu().squeeze().numpy(), c.cpu().squeeze().numpy(), l.cpu().numpy(), fn=cal_pesq)
                pbar_dict = OrderedDict({
                    'val_loss': loss.item(),
                    'val_pesq': pesq_score.item(),
                })
                pbar.set_postfix(pbar_dict)

                total_loss += loss.item()
                total_pesq += pesq_score.item()
                del noisy, clean, n, c, ewav, ereal, eimg, creal, cimg

            if scheduler is not None:
                scheduler.step(total_pesq / num_test_data)

            writer.add_scalar('Loss/valid', total_loss / num_test_data, epoch)
            writer.add_scalar('PESQ/valid', total_pesq / num_test_data, epoch)

            # checkpointing
            curr_pesq = total_pesq / num_test_data
            curr_loss = total_loss / num_test_data
            if  curr_pesq > best_pesq:# or ((epoch+1)%5)==0:
                best_pesq = curr_pesq
                save_path = os.path.join(ckpt_path, 'model_best.ckpt')
                print(f'Saving checkpoint to {save_path}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': se_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss / num_test_data,
                    'pesq': total_pesq / num_test_data
                }, save_path)
        
        print("Val_PESQ_avg: ", curr_pesq)
    writer.flush()
    writer.close()


