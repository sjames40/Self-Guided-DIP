import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Sequence, Tuple, Union
import math
import time
from torch.utils.data.dataset import Dataset
from torch.nn import init
import math
import scipy
import scipy.linalg
import h5py
import sys
import os
import torch
from util.util import generate_mask_alpha, generate_mask_beta
import scipy.ndimage
from util.util import fft2, ifft2, cplx_to_tensor, complex_conj, complex_matmul, absolute
import h5py
import glob
from models import networks

def convert_2chan_into_abs_2(img):
    """
    Converts a 2-channel image (real, imag) into a complex tensor.
    """
    img_real = img[0][0]
    img_imag = img[0][1]
    img_complex = torch.complex(img_real, img_imag)
    return img_complex

def make_data_list(file_path,file_array):
    """
    Loads data from a list of filenames into a list in memory.
    """
    file_data = []
    for i in range(len(file_array)):
        data_file = file_array[i]
        data_from_file = np.load(os.path.join(file_path,data_file),'r')
        file_data.append(data_from_file)
    return file_data

def make_vdrs_mask(N1,N2,nlines,init_lines,seed=0):
    """
    Generates a Variable Density Random Sampling (VDRS) mask.
    
    Args:
        N1, N2: Dimensions of the mask.
        nlines: Total number of lines to sample (approximate/target).
        init_lines: Number of fully sampled central lines (Auto-Calibration Signal).
        seed: Random seed for reproducibility.
    """
    mask_vdrs=np.zeros((N1,N2),dtype='bool')
    low1=(N2-init_lines)//2
    low2=(N2+init_lines)//2
    mask_vdrs[:,low1:low2]=True # Fully sample the center (ACS region)
    nlinesout=(nlines-init_lines)//2
    rng = np.random.default_rng(seed)
    # Randomly sample lines outside the center
    t1 = rng.choice(low1, size=nlinesout, replace=False)
    t2 = rng.choice(np.arange(low2, N2-1), size=nlinesout, replace=False)
    mask_vdrs[:,t1]=True; mask_vdrs[:,t2]=True
    return mask_vdrs

# --- Configuration and Data Loading ---
Kspace_data_name = 'Self-Guided-DIP/NEW_KSPACE' # This is the path to the kspace data
kspace_array = os.listdir(Kspace_data_name)
kspace_array = sorted(kspace_array)

kspace_data = []

number =0

# Select specific file index to load
index = 331+number                
kspace_file = kspace_array[index]
kspace_data_from_file = np.load(os.path.join(Kspace_data_name,kspace_file),'r')
kspace_data.append(kspace_data_from_file)

mask_vali =[]

class nyumultidataset(Dataset): # model data loader
    """
    Dataset class for loading and processing MRI data.
    Handles k-space data, sensitivity maps, undersampling masks, and generates
    input/target pairs for the reconstruction model.
    """
    def  __init__(self ,kspace_data,mask_data):
        self.A_paths = kspace_data
        self.A_size = len(self.A_paths)
        self.mask_path = mask_data
        self.nx = 640
        self.ny = 368

    def __getitem__(self, index):
        A_temp = self.A_paths[index]
        
        # --- 1. Load and Normalize Data ---
        # Load sensitivity maps (s) and k-space (k) data
        # Data is normalized by 32767.0 (assuming 16-bit signed integer range)
        s_r = A_temp['s_r']/ 32767.0 
        s_i = A_temp['s_i']/ 32767.0 
        k_r = A_temp['k_r']/ 32767.0
        k_i = A_temp['k_i']/ 32767.0 
        ncoil, nx, ny = s_r.shape
        
        # --- 2. Generate Mask ---
        # Create Variable Density Random Sampling mask
        mask_in = make_vdrs_mask(nx,ny,int(ny*0.25),int(ny*0.08))
        
        # --- 3. Prepare Data Structures ---
        k_np = np.stack((k_r, k_i), axis=0)
        
        # Crop sensitivity maps to center 320x320 region (from 640x368)
        # Note: Center indices are calculated relative to nx/2 and ny/2
        s_np = np.stack((s_r[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160],
                         s_i[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]), axis=0)
        s_np_no_crop = np.stack((s_r,
                         s_i), axis=0)                 
        
        # Prepare mask tensor: replicate mask for 2 channels (real/imag) and crop to center
        mask = torch.tensor(np.repeat(mask_in[np.newaxis, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160], 2, axis=0), dtype=torch.float32)
        
        # --- 4. Image Reconstruction and Forward Model ---
        # Convert k-space to tensor: (ncoil, 2, nx, ny)
        A_k = torch.tensor(k_np, dtype=torch.float32).permute(1, 0, 2, 3)
        # Inverse FFT to get image domain: (ncoil, 2, nx, ny) -> (ncoil, 2, nx, ny)
        A_I = ifft2(A_k.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Crop image to center region
        A_I = A_I[:, :, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]
        
        ##A_s is the sensitive map 
        A_s = torch.tensor(s_np, dtype=torch.float32).permute(1, 0, 2, 3)
        A_s_no_crop = torch.tensor(s_np_no_crop, dtype=torch.float32).permute(1, 0, 2, 3)
        
        # Calculate Sum of Squares (SOS) for normalization reference
        SOS = torch.sum(complex_matmul(A_I, complex_conj(A_s)),dim=0)
        # Normalize image by the max magnitude of SOS
        A_I = A_I/torch.max(torch.abs(SOS)[:])
        
        # Forward projection: Normalized Image -> K-space
        A_k2 = fft2(A_I.permute(0,2,3,1)).permute(0,3,1,2)
        kreal = A_k2
        
        # Initialize System Operator (Sensitivity Encoding)
        AT = networks.OPAT2(A_s)
        
        # Generate Undersampled Image (Iunder) using the mask
        Iunder = AT(kreal, mask)
        # Generate Fully Sampled Ground Truth (Ireal)
        Ireal = AT(kreal, torch.ones_like(mask))
        
        return  Iunder, Ireal, A_s, mask, mask_in,A_s_no_crop, A_k
     
       
    def __len__(self):
        return len(self.A_paths)

    
# --- Create Data Loader ---
test_clean_paths = kspace_data
mask_test_paths = mask_vali
test_dataset = nyumultidataset(test_clean_paths,mask_test_paths)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=False)
