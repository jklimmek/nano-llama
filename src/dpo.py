import contextlib
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import NanoLlama
from utils import *


class DPODataset(Dataset):
    """Each instance in dataset is padded to pre-defined length and contains special tokens.

    Tokens of each sample: 
    [TITLE] <Example title here> [CONTEXT] <Accepted article tokens> [END-OF-TEXT] [PADDING] 
    [TITLE] <Example title here> [CONTEXT] <Rejected article tokens> [END-OF-TEXT] [PADDING] 
    """
    def __init__(self, data_path, device):
        super().__init__()
