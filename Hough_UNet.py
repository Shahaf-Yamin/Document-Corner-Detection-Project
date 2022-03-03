# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:33:26 2022

@author: yoell
"""

import os                                   
from PIL import Image                        
import PIL.Image as Image
import torch  
import torch.nn.functional as F          
import numpy as np                    
import matplotlib.pyplot as plt                 
from Robust_Hough import *
from unet.unet_model import UNet
from utils.utils import *


device = torch.device('cpu')




    
def example(path_to_images, bg=1,img=3):
    
    TransformsMapping = {'Resize': Resize, 'GaussianBlur': ourGaussianBlur, 'Normalize': Normalize, 'RandomCropNearCorner': RandomCropNearCorner}
    FILTER_SIZE = 5 #Gaussian Filter
    SIZE = 512 #image size 512X512
    FRAMES_PER_VIDEO = 5 # Single Sample from each video
    test_set_bg = DocumentDatasetMaskSegmentation(TransformsMapping, Path=os.path.join(path_to_images,"background0")+str(bg), frames_per_video=FRAMES_PER_VIDEO, Transforms={'GaussianBlur': FILTER_SIZE, 'Normalize': (0, 1)}, Size=SIZE)

    w_path = "Unet512x512_checkpoint_filter_5x5.pth"
    pretrained_unet = UNet(3, 2)
    pretrained_unet.load_state_dict(torch.load(w_path, map_location=torch.device('cpu')))

    data_example = test_set_bg[img]
    # precalculated mean and std over the full train set
    train_mean = 151.84097687205238 
    train_std = 43.61468699572161
    img = data_example['image']
    gt_mask = data_example['mask'].T
    norm_img = (img - train_mean) / train_std
    pred_mask = pretrained_unet(norm_img.unsqueeze(dim=0).to(device)).detach()
    pred_mask = F.one_hot(pred_mask.argmax(dim=1), pretrained_unet.n_classes).permute(0, 3, 1, 2).float().numpy()[0,1,:,:].T
    fig, ax = plt.subplots(1, 3)
    fig.suptitle('UNet Performance Example')
    ax[0].set_title('Input Image')
    ax[0].imshow(img.transpose(0,-1).int())
    ax[1].set_title('Ground Truth Mask')
    ax[1].imshow(gt_mask)
    ax[2].set_title('Predicted Mask')
    ax[2].imshow(pred_mask)
    
    edge_img = DerivativeEdges(pred_mask)
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Edges Extractor')
    ax[0].set_title('Predicted Mask')
    ax[0].imshow(pred_mask)
    ax[1].set_title('Edged Mask')
    ax[1].imshow(edge_img)
    
    corners_estimation_closest_point, corners_estimation, H, T, R, hough_peaks, estimated_peaks = e2e_algorithm(pred_mask, numPeaks=20)
    plot_hough(H, estimated_peaks, hough_peaks, T, R)
    
    plot_img_and_mask(img.transpose(0,-1).int(), gt_mask, corners_estimation)
    
    plot_img_and_mask(img.transpose(0,-1).int(), gt_mask, corners_estimation_closest_point)
    
if __name__ == "__main__":
    path_to_images=r'C:\Users\yoell\Desktop\Electrical_Engineering\Master_Degree\2ndYear\IntroToDL\Recursive-CNNs-Pytorch-RecursiveCNN\Test_set\Test\Test'
    example(path_to_images, bg=1, img=3)
