from dataset import get_data_transforms
import torch
from torchvision import transforms
from dataset import MVTecDataset, MVTecTestDataset
from torch.nn import functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
from numpy import ndarray
import pandas as pd
from skimage import measure
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
import os
import tifffile as tiff
import cv2
import torch
import torch.nn as nn
from models import vit_encoder
from models.uad import INP_Former, Shift_Former, INPFA_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block, Prototype_Block_self
import torch.multiprocessing as mp
from functools import partial
import argparse


def calculate_segF1_max(segmentations, masks_gt):
        masks_gt = [mask.squeeze().cpu().numpy() if isinstance(mask, torch.Tensor) else np.squeeze(mask)
            for mask in masks_gt]

        thresholds = np.linspace(0, 1, 101)
        best_f1 = 0
        best_threshold = 0
        binary_masks = []

        for threshold in thresholds:
            f1_scores = []
            for seg, mask_gt in zip(segmentations, masks_gt):
                if seg.shape != mask_gt.shape:
                    raise ValueError(f"Shape mismatch: seg {seg.shape}, mask_gt {mask_gt.shape}")
                pred_mask = (seg >= threshold).astype(np.uint8)
                tp = np.sum((pred_mask == 1) & (mask_gt == 1))
                fp = np.sum((pred_mask == 1) & (mask_gt == 0))
                fn = np.sum((pred_mask == 0) & (mask_gt == 1))
                
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
                f1_scores.append(f1)
            
            mean_f1 = np.mean(f1_scores)
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_threshold = threshold

        for seg in segmentations:
            binary_masks.append((seg >= best_threshold).astype(np.uint8))

        return best_threshold, binary_masks, best_f1
def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        # df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        new_row = pd.DataFrame([{"pro": np.mean(pros), "fpr": fpr, "threshold": th}])
        df = pd.concat([df, new_row], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def cal_anomaly_maps(fs_list, ft_list,  out_size_h, out_size_w):

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        # mse_map = torch.mean((fs-ft)**2, dim=1)
        # a_map = mse_map
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=(out_size_h, out_size_w), mode='bilinear', align_corners=True)
        # a_map = a_map[0, 0, :, :]
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * np.math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def evaluation(item, args):
    # model_path = os.path.join(args.model_path, 'models') + '/inp-former++_' + item + '.pth'
    model_path = os.path.join(args.model_path, 'models') + '/vitrd_' + item + '.pth'
    test_path = os.path.join(args.dataset_path, item)
    root_path = args.output_path

    resize_ratio = 0.5
    encoder_name = 'dinov2reg_vit_base_14'
    INP_num = 6
    device = f'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(model_path)
    data_transform, gt_transform = get_data_transforms(resize_ratio)
    best_threshold_list_p = {'can': 0.203, 'fabric': 0.122, 'fruit_jelly': 0.177, 'rice': 0.153, 'sheet_metal': 0.216, 'vial': 0.219, 'wallplugs': 0.195, 'walnuts': 0.134}
    best_threshold_list_m = {'can': 0.203, 'fabric': 0.122, 'fruit_jelly': 0.177, 'rice': 0.153, 'sheet_metal': 0.216, 'vial': 0.219, 'wallplugs': 0.155, 'walnuts': 0.134}

    
    test_data = MVTecTestDataset(root=test_path, transform=data_transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)


    checkpoint = torch.load(model_path, map_location=device)

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)
    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."
    
    Bottleneck = []
    INP_Guided_Decoder = []
    INP_Extractor = []

    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)

    INP = nn.ParameterList(
                    [nn.Parameter(torch.randn(INP_num, embed_dim))
                     for _ in range(1)])
    
    for i in range(1):
        blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Extractor.append(blk)
    INP_Extractor = nn.ModuleList(INP_Extractor)

    for i in range(8):
        # blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
        #                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Guided_Decoder.append(blk)
    INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

    model = INPFA_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                             target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                             fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    with torch.no_grad():
        for img, name, img_type in test_dataloader:
            
            
            name = name[0]
            img_type = img_type[0]
            ext = name.split('.')[-1]

            img = img.to(device)
            en, de,  g_loss = model(img)

            _, _, h, w = img.shape
            anomaly_map, _ = cal_anomaly_maps(en, de, h,w)
            anomaly_map = anomaly_map.cpu().detach().numpy()
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            anomaly_map= anomaly_map.squeeze()

            os.makedirs(os.path.join(root_path, 'anomaly_images', item, img_type, os.path.dirname(name)), exist_ok=True)
            os.makedirs(os.path.join(root_path, 'anomaly_images_thresholded', item, img_type, os.path.dirname(name)), exist_ok=True)
            tiff.imwrite(os.path.join(root_path, 'anomaly_images', item, img_type, name.replace(ext, 'tiff')), anomaly_map.astype(np.float16))
            if img_type == 'test_private':
                threshold = best_threshold_list_p[item]
            elif img_type == 'test_private_mixed':
                threshold = best_threshold_list_m[item]
            
            # _, thred = cv2.threshold(anomaly_map, threshold, 255, cv2.THRESH_BINARY)
            thred = np.where(anomaly_map > threshold, 255, 0).astype(np.uint8)
            if item == 'fabric':
                if img_type == 'test_private':
                    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))         
                    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
                else:
                    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91))         
                    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (111, 111))
            elif item == 'sheet_metal':
                kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))         
                kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
            else:
                kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))         
                kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

            dilated = cv2.dilate(thred, kernel1, iterations=1)
            fill_holes = dilated.copy()
            contours, _ = cv2.findContours(fill_holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(fill_holes, [cnt], 0, 255, thickness=cv2.FILLED)
            thred = cv2.erode(fill_holes, kernel2, iterations=1)

            print(threshold, thred.shape, anomaly_map.shape)
            cv2.imwrite(os.path.join(root_path, 'anomaly_images_thresholded', item, img_type, name.replace(ext, 'png')), thred.astype(np.uint8))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    item_list = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
    
    
    for item in item_list:
        print(item)
        evaluation(item, args)
        

if __name__ == '__main__':
    main()