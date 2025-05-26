from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torch.utils.data import Dataset
import numpy as np
import pdb
from torch.utils.data import Dataset
from torchvision import transforms as T
import imgaug.augmenters as iaa
import cv2
import math


augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50,50),per_channel=True),
            iaa.Solarize(0.5, threshold=(32,128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
            ]
rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

# def lerp_np(x,y,w):
#     fin_out = (y-x)*w + x
#     return fin_out

def rand_perlin_2d_np(shape, res, fade=lambda t: 6*t**5-15*t**4+10*t**3):
    res = (res, res) if isinstance(res, int) else res
    d = (
        math.ceil(shape[0]/res[0]),  # 关键修改：向上取整
        math.ceil(shape[1]/res[1])
    )
    
    # 生成精确网格
    x = np.linspace(0, res[0], shape[0], endpoint=False)
    y = np.linspace(0, res[1], shape[1], endpoint=False)
    grid = np.stack(np.meshgrid(x, y, indexing='ij'), axis=-1) % 1
    
    # 生成梯度场
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    
    # 修正梯度平铺函数参数
    def tile_grads(slice_x, slice_y):
        grad_slice = gradients[slice_x, slice_y]
        # 平铺后立即裁剪到目标尺寸
        tiled = np.repeat(np.repeat(grad_slice, d[0], axis=0), d[1], axis=1)
        return tiled[:shape[0], :shape[1]]  # 新增强制裁剪
    
    # 计算四角贡献
    def dot(shift_x, shift_y):
        # 修正：传递两个slice对象
        grad_slice = tile_grads(
            slice(shift_x, None),  # x方向切片
            slice(shift_y, None)   # y方向切片
        )[:shape[0], :shape[1]]
        shifted_grid = (grid - np.array([shift_x, shift_y])) % 1
        return (shifted_grid * grad_slice).sum(axis=-1)

    n00 = dot(0, 0)
    n10 = dot(1, 0)
    n01 = dot(0, 1)
    n11 = dot(1, 1)
    
    # 插值计算
    t = fade(grid)
    top = lerp_np(n00, n10, t[..., 0])
    bottom = lerp_np(n01, n11, t[..., 0])
    return math.sqrt(2) * lerp_np(top, bottom, t[..., 1])

def lerp_np(a, b, t):
    return a + t * (b - a)

def transform_image(image, fore_mask, anomaly_source_paths, idx=None):
    anomaly_source_idx = torch.randint(0, len(anomaly_source_paths), (1,)).item()
    anomaly_source_path = anomaly_source_paths[anomaly_source_idx]
    # image = image.permute(1,2,0).numpy()
    if fore_mask is not None:
        fore_mask = fore_mask.permute(1,2,0).numpy()
    # normalize the image to 0.0~1.0
    augmented_image, anomaly_mask, has_anomaly = augment_image(image, anomaly_source_path, fore_mask)
    # augmented_image = np.transpose(augmented_image, (2, 0, 1))
    # image = np.transpose(image, (2, 0, 1))
    anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
    if  False:
        save_path = '/data/zhangxin/zhang/INP-Former-main/result/0.5-all/seg/Aimg'
        os.makedirs(save_path, exist_ok=True)
        img_save_path = os.path.join(save_path, f"image_{idx}.png")
        mask_save_path = os.path.join(save_path, f"mask_{idx}.png")
        cv2.imwrite(img_save_path, augmented_image)
        mask_to_save = np.squeeze(anomaly_mask) * 255  # shape: (H, W)
        cv2.imwrite(mask_save_path, mask_to_save.astype(np.uint8))
    return augmented_image, anomaly_mask, has_anomaly

def get_center_mask(shape, center_ratio=0.5):
    """
    shape: (H, W)
    center_ratio: 中心区域占比（如 0.5 表示宽高各取中间 50%）
    """
    h, w = shape
    center_h = int(h * center_ratio)
    center_w = int(w * center_ratio)
    start_h = (h - center_h) // 2
    start_w = (w - center_w) // 2
    mask = np.zeros(shape, dtype=np.float32)
    mask[start_h:start_h+center_h, start_w:start_w+center_w] = 1
    return mask[..., np.newaxis]

def augment_image(image, anomaly_source_path, fore_mask):
    aug = randAugmenter()
    min_perlin_scale = 2
    max_perlin_scale = 5
    height, width, c = image.shape
    # print(width, height)
    anomaly_source_img = cv2.imread(anomaly_source_path)
    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(width, height))

    anomaly_img_augmented = aug(image=anomaly_source_img)
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0])
    
    if fore_mask is not None:
        while True:
            perlin_noise = rand_perlin_2d_np((height, width), (perlin_scalex, perlin_scaley))
            perlin_noise = rot(image=perlin_noise)
            threshold = 0.5
            # modify
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_thr = np.expand_dims(perlin_thr, axis=2)
            perlin_thr = perlin_thr * fore_mask
            # pdb.set_trace()
            if perlin_thr.sum() > 4:
                break
    else:
        # print((height, width), (perlin_scalex, perlin_scaley))
        perlin_noise = rand_perlin_2d_np((height, width), (perlin_scalex, perlin_scaley))
        perlin_noise = rot(image=perlin_noise)
        threshold = 0.8
        # modify
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

    img_thr = anomaly_img_augmented

    center_mask = get_center_mask((height, width), center_ratio=0.5)
    perlin_thr = perlin_thr * center_mask

    beta = torch.rand(1).numpy()[0] * 0.8
    augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
        perlin_thr)
    
    augmented_image = augmented_image.astype(np.float32)
    msk = (perlin_thr).astype(np.float32)
    augmented_image = msk * augmented_image + (1-msk)*image  # This line is unnecessary and can be deleted
    has_anomaly = 1.0

    no_anomaly = torch.rand(1).numpy()[0]
    if no_anomaly > 0.7:
        image = image.astype(np.float32)
        return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
    else:  # 0.7概率产生异常
        augmented_image = augmented_image.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1-msk)*image  # This line is unnecessary and can be deleted
        has_anomaly = 1.0
        if np.sum(msk) == 0:
            has_anomaly=0.0
        return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

def randAugmenter():
    aug_ind = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
    aug = iaa.Sequential([augmenters[aug_ind[0]],
                        augmenters[aug_ind[1]],
                        augmenters[aug_ind[2]]]
                            )
    return aug