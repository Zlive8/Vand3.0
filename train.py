import torch
import torch.nn as nn
import numpy as np
import random
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from optimizers import StableAdamW
from utils import evaluation,WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, global_kl_hm_adaptive

# Dataset-Related Modules
from dataset import get_data_transforms, MVTecDataset, AugMixDatasetMVTec
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from utils import augmentation

# Model-Related Modules
from models import vit_encoder
from models.uad import INP_Former, Self_Former, Shift_Former, INPFA_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block, Prototype_Block_self
import torch.multiprocessing as mp
import datetime
import logging
from torchvision import transforms

warnings.filterwarnings("ignore")

def train_shift(_class_, args, gpu_id, result_queue):
    print(_class_)
    print(f"Training class {_class_} on GPU {gpu_id}")
    resize_ratio = 0.5
    batch_size = 8
    epochs = 200
    INP_num = 6
    encoder_name = 'dinov2reg_vit_base_14'
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(resize_ratio)
    resize_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.resize((
            ((int(img.width*resize_ratio) + 7) // 14) * 14, 
            ((int(img.height*resize_ratio) + 7) // 14) * 14
        )))
    ])
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train),
    ])


    train_path = os.path.join(args.dataset_path, _class_, 'train')
    test_path = os.path.join(args.dataset_path, _class_, 'test_public')
    ckp_path = os.path.join(args.save_path, 'models/inp-former++_'+_class_+'.pth')
    print(ckp_path)

    train_data = ImageFolder(root=train_path, transform=resize_transform)
    train_data = AugMixDatasetMVTec(train_data, preprocess, _class_)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    
    # Adopting a grouping-based reconstruction strategy similar to Dinomaly
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Encoder info
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

    # Model Preparation
    Bottleneck = []
    INP_Guided_Decoder = []
    INP_Extractor = []

    # bottleneck
    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)

    # INP
    INP = nn.ParameterList(
                    [nn.Parameter(torch.randn(INP_num, embed_dim))
                     for _ in range(1)])

    # INP Extractor
    for i in range(1):
        blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Extractor.append(blk)
    INP_Extractor = nn.ModuleList(INP_Extractor)

    # INP_Guided_Decoder
    for i in range(8):
        blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Guided_Decoder.append(blk)
    INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

    model = INPFA_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                             target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                             fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
    model = model.to(device)

    # Model Initialization
    trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    # define optimizer
    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4, total_iters=epochs*len(train_dataloader),
                                        warmup_iters=100)

    # Train
    for epoch in range(epochs):
        model.train()
        loss_list = []
        for img in train_dataloader:
            img = img.to(device)
            en, de,  g_loss = model(img)
            
            loss_cosin = global_cosine_hm_adaptive(en, de, y=3)  + 0.2 * g_loss
            loss_kl = global_kl_hm_adaptive(en, de, y=3)
            loss = loss_cosin * 0.9 + loss_kl * 0.1

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()
        print('class:{}, epoch [{}/{}], loss:{:.4f}'.format(_class_, epoch + 1, epochs, np.mean(loss_list)))
        # logging.info('class:{}, epoch [{}/{}], loss:{:.4f}'.format(_class_, epoch + 1, epochs, np.mean(loss_list)))
        # logging.getLogger().handlers[0].flush()
        if (epoch + 1) % 10 == 0:
                results = evaluation(model, test_dataloader, device, max_ratio=0.01)
                auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, best_thre, aupro_px = results
                print('Pixel Auroc:{:.3f},  Pixel Aupro:{:.3},  Pixel SegF1 Max:{:.3}'.format(auroc_px,  aupro_px, f1_px))
                # logging.info('class:{}, Pixel Auroc:{:.3f},  Pixel Aupro:{:.3f},  Pixel SegF1 Max:{:.3},  Best_thre:{:.3}'.format(_class_, auroc_px, aupro_px, f1_px, best_thre))
                # logging.getLogger().handlers[0].flush()
                torch.save(model.state_dict(), ckp_path)
    result_queue.put((auroc_px, aupro_px, f1_px))

def train_multiple_classes(classes, args):
    mp.set_start_method('spawn', force=True)
    target_gpus = [4, 5, 6, 7]
    processes = []
    results = []
    result_queue = mp.Queue()

    for i, _class_ in enumerate(classes):
        gpu_id = target_gpus[i % len(target_gpus)]
        process = mp.Process(target=train_shift, args=(_class_, args, gpu_id, result_queue))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    while not result_queue.empty():
        results.append(result_queue.get())

    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='./saved_models')


    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.join(args.save_path, 'models'))

    # item_list = ['can', 'sheet_metal', 'fruit_jelly', 'vial', 'fabric', 'rice', 'wallplugs', 'walnuts']
    item_list = ['can', 'sheet_metal', 'fruit_jelly', 'vial']
    num_classes = len(item_list)
    batch_size = 4

    setup_seed(111)
    auroc_list = []
    aupro_list = []
    best_f1_list = []

    for i in range(0, num_classes, batch_size):
        selected_classes = item_list[i:i + batch_size]
        results = train_multiple_classes(selected_classes, args)
        for result in results:
            auroc_list.append(result[0])
            aupro_list.append(result[1])
            best_f1_list.append(result[2])

    avg_auroc = np.mean(auroc_list)
    avg_aupro = np.mean(aupro_list)
    avg_best_f1 = np.mean(best_f1_list)

    print(f'Avg Pixel Auroc: {avg_auroc:.3f}')
    print(f'Avg Pixel Aupro: {avg_aupro:.3f}')
    print(f'Avg Pixel SegF1 Max: {avg_best_f1:.3f}')
    # logging.info(f'Avg Pixel Auroc: {avg_auroc:.3f}')
    # logging.info(f'Avg Pixel Aupro: {avg_aupro:.3f}')
    # logging.info(f'Avg Pixel SegF1 Max: {avg_best_f1:.3f}')

if __name__ == '__main__':
    main()  
