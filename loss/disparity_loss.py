import numpy as np
import torch

def norm(batch):
    batch = np.asarray(batch)
    b, h, w, c = batch.shape
    norm = np.zeros([b, c, h, w], "float32")
    r = batch[:, :, :, 0]
    g = batch[:, :, :, 1]
    b = batch[:, :, :, 2]
    norm[:, 0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    norm[:, 1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    norm[:, 2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return norm

def preprocess(batch, crop_height, crop_width):
    batch = norm(batch)
    b, c, h, w = batch.shape
    if h <= crop_height and w <= crop_width:
        temp = np.zeros([b, c, crop_height, crop_width], "float32")
        temp[:, :, crop_height - h: crop_height, crop_width - w: crop_width] = batch.copy()
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp = batch[:, :, start_y: start_y + crop_height, start_x: start_x + crop_width].copy()
    
    return torch.from_numpy(temp).float(), h, w
