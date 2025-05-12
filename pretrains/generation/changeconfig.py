import torch
from collections import OrderedDict
import sys

data = torch.load('checkpoint.pth')
config = data['config']

config.eeg_signals_path = "./datasets/eeg_5_95_std.pth"
config.splits_path = "./datasets/block_splits_by_image_single.pth"
config.pretrain_mbm_path = "./pretrains/eeg-pretrain/checkpoint-eeg-500.pth"
config.pretrain_gm_path = "./pretrains"

torch.save(data, 'checkpoint2.pth')
