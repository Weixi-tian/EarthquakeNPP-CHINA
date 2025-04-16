import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.data import SlidingWindowWrapper
from download_data import download
from loguru import logger

npz = np.load('/home/tianweixi/Programe/EarthquakeNPP_CSEP_China/Experiment/AutoSTPP_new_version/data/spatiotemporal/CENC_M40_deepstpp_mag.npz', allow_pickle=True)


sliding_test = SlidingWindowWrapper(npz['test'], normalized=True, device="cpu")

datatest_loader = DataLoader(sliding_test, batch_size=128)


for batch_idx, batch in enumerate(datatest_loader):
    print(batch_idx)
    print('st_x',batch[1])
    print(batch[0].shape)


