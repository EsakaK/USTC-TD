import numpy as np
import os
import glob
import openpyxl
from pytorch_msssim import ms_ssim

import concurrent.futures
import json
import re
import multiprocessing
import argparse
from src.utils.functional import filter_dict
from src.utils.video_reader import YUVReader
from src.utils.functional import ycbcr2rgb, np_image_to_tensor, PSNR, mse2PSNR, calc_msssim, calc_msssim_rgb
import torch


def calc_one_sequence(orig_yuv_path, recon_yuv_path, width_lr=1920, height_lr=1080, frame_to_encode=96, device='cuda:0'):
    yuv_reader_BL_gt = YUVReader(orig_yuv_path, width_lr, height_lr)
    yuv_reader_BL_recon = YUVReader(recon_yuv_path, width_lr, height_lr)
    
    BL_rgb_psnr = 0
    BL_rgb_msssim = 0
    for frame_idx in range(frame_to_encode):
        y_BL, uv_BL = yuv_reader_BL_gt.read_one_frame()  #
        yuv_BL = np.concatenate([y_BL, uv_BL])
        rgb_BL = ycbcr2rgb(np_image_to_tensor(yuv_BL)).to(device).detach()

        y_rec_BL, uv_rec_BL = yuv_reader_BL_recon.read_one_frame()
        yuv_rec_BL = np.concatenate([y_rec_BL, uv_rec_BL])#concatenate((arr1, arr2), axis = None) 
        rec_BL = ycbcr2rgb(np_image_to_tensor(yuv_rec_BL)).to(device).detach()

        # RGB PSNR, MS-SSIM
        BL_rgb_psnr += PSNR(rgb_BL, rec_BL)
        BL_rgb_msssim += calc_msssim_rgb(rgb_BL[0].cpu().numpy(), rec_BL[0].cpu().numpy(), data_range=1.0)
        
        print(f'{frame_idx} / {frame_to_encode}')
    
    avg_psnr = BL_rgb_psnr / frame_to_encode
    avg_msssim = BL_rgb_msssim / frame_to_encode
    yuv_reader_BL_gt.close()
    yuv_reader_BL_recon.close()
    
    return avg_psnr, avg_msssim   



# 原始视频存储位置
orig_yuv_dir = r'\\192.168.9.1\share\data\liaojunqi\0dataset\VCIP2023\yuv444'
# 重建视频存储位置
recon_yuv_dir = r'E:\Otherprojects\iVC dataset\anchor\video_recs'
# 重建yuv任务名，用于区分出当前任务的yuv文件
profile_name = 'VTM132_XHH_HM_444'#'VTM132_XHH_444'
# 序列名
seq_names = ['USTC_Badminton', 'USTC_BasketballDrill', 'USTC_BasketballPass', 'USTC_BicycleDriving', 'USTC_Dancing', 'USTC_FourPeople', 'USTC_ParkWalking', 'USTC_Running', 'USTC_ShakingHands', 'USTC_Snooker']
# QP
QPs = [22, 27, 32, 37, 42]
# excel data
excel_wb = openpyxl.Workbook()
test_sheet = excel_wb.active
excel_data = [['Seq Name', 'QP', 'RGB-PSNR', 'RGB-MSSSIM']]

# 遍历序列，计算RGBPSNR
for seq_name in seq_names:
    for QP in QPs:
        orig_yuv_filename = os.path.join(orig_yuv_dir, f'{seq_name}.yuv')
        recon_yuv_filename = glob.glob(f'{recon_yuv_dir}/*{seq_name[4:]}*{profile_name}_Q{QP}.yuv')[0]
        recon_yuv_filename = os.path.join(recon_yuv_dir, recon_yuv_filename)
        
        curr_psnr, curr_msssim = calc_one_sequence(orig_yuv_filename, recon_yuv_filename)
        
        excel_data.append([seq_name, QP, curr_psnr, curr_msssim])

for line in excel_data:
    test_sheet.append(line)
excel_wb.save(filename='./RGB_PSNR.xlsx')