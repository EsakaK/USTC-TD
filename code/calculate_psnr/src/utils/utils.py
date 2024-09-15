#----------------------------------------------------------------------------------#
import os
import platform
from tqdm import tqdm
import shutil
from PIL import Image
from torchvision import transforms
from .core import imresize
from torch import cuda
from .video_reader import YUVReader
from .video_writer import YUVWriter
import torch

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


def read_batch(d_path, batch_names) -> torch.Tensor:
    torch_imgs = []
    for img_name in batch_names:
        img_path = os.path.join(d_path, img_name)
        img = Image.open(img_path)
        torch_imgs.append(to_tensor(img))
    batch_data = torch.stack(torch_imgs, 0)
    return batch_data


def write_batch(new_d_path, batch_names, batch_data):
    if not os.path.exists(new_d_path):
        os.mkdir(new_d_path)
    for index, img_name in enumerate(batch_names):
        img_path = os.path.join(new_d_path, img_name)
        torch_img = batch_data[index]
        pil_img = to_pil(torch_img)
        pil_img.save(img_path)


def resize_batch(batch_data: torch.Tensor, size=None) -> torch.Tensor:
    if cuda.is_available():
        batch_data = batch_data.cuda()
    batch_data_ds = imresize(batch_data, sizes=size, kernel='cubic')
    return batch_data_ds


def resize_yuv(yuv_path, new_yuv_path, size=None, ori_size = None, frame_num=96, flag = True):
    yuv_reader = YUVReader(yuv_path, width=ori_size[1], height=ori_size[0])
    yuv_writer = YUVWriter(new_yuv_path, width=size[1], height=size[0])
    Y_list = []
    UV_list = []
    for i in range(frame_num):
        y, uv = yuv_reader.read_one_frame()  # torch Tensor
        Y_list.append(y)
        UV_list.append(uv)
    yuv_reader.close()

    batch_size = 32
    sized_Y = torch.zeros((frame_num,1,size[0],size[1]))
    sized_UV = torch.zeros((frame_num,2,size[0]//2,size[1]//2))
    for b_num in range(frame_num//batch_size):
        Y_batch = Y_list[b_num * batch_size:(b_num+1)*batch_size]
        UV_batch = UV_list[b_num * batch_size:(b_num + 1) * batch_size]
        with torch.no_grad():
            Y_batch = torch.stack(Y_batch).cuda(3)
            UV_batch = torch.stack(UV_batch).cuda(3)
            if flag: # need resize
                sized_Y_sequence = imresize(Y_batch,sizes=size,kernel='cubic')
                sized_UV_sequence = imresize(UV_batch, sizes=(size[0]//2, size[1]//2), kernel='cubic')
            else:
                sized_Y_sequence = Y_batch
                sized_UV_sequence = UV_batch
            sized_Y[b_num * batch_size:(b_num+1)*batch_size,:,:,:] = sized_Y_sequence
            sized_UV[b_num * batch_size:(b_num + 1) * batch_size, :, :, :] = sized_UV_sequence
            # write to byte
    print('resize over!')
    for i in range(frame_num):
        y = sized_Y[i].numpy()
        uv = sized_UV[i].numpy()
        yuv_writer.write_one_frame(y=y,uv=uv)
    yuv_writer.close()
    print(new_yuv_path)






def resize_one_directory(d_path, new_d_path, size=None, frame_num=96, batch_size=2):
    img_names = os.listdir(d_path)
    img_names = list(filter('x1'.__ne__, img_names))
    img_names = list(filter('x1_5'.__ne__, img_names))
    img_names = list(filter('x2'.__ne__, img_names))
    img_names = list(filter('x4'.__ne__, img_names))
    img_names.sort()
    if size is None:
        for i in range(0, min(frame_num, len(img_names))):
            if platform.system().lower() == 'windows':
                os.system(f"copy {d_path}\\{img_names[i]} {new_d_path}")
            elif platform.system().lower() == 'linux':
                os.system(f"cp {d_path}/{img_names[i]} {new_d_path}/")
        return
    for i in range(0, min(frame_num, len(img_names)), batch_size):
        batch_names = []
        for j in range(batch_size):
            if i + j >= frame_num:
                break
            batch_names.append(img_names[i + j])
        batch_data = read_batch(d_path, batch_names)
        ds_batch_data = resize_batch(batch_data, size).clamp_(0, 1)
        write_batch(new_d_path, batch_names, ds_batch_data)


def compose_one_directory(image_path, fps, yuv_path, resolution, pix_fmt):
    cd_command = f'cd {image_path}'
    compose_command = f'Z:/user/bianyifan/ffmpeg/bin/ffmpeg.exe -i im%5d.png -pix_fmt {pix_fmt}p -s {resolution} {yuv_path}.yuv'
    full_command = cd_command + ' && ' + compose_command
    os.system(full_command)
    print(yuv_path + '.yuv')


def decompose_one_yuv(d_path, resolution, pix_fmt):
    mkdir_command = f"mkdir {d_path}"
    yuv2png_command = f"Z:/user/bianyifan/ffmpeg/bin/ffmpeg.exe -pix_fmt {pix_fmt}p -s {resolution} -i {d_path}.yuv -f image2 {d_path}/im%05d.png"
    full_command = mkdir_command + ' && ' + yuv2png_command
    os.system(full_command)


def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()


def psnr_between_pic(img1_path, img2_path):
    img1 = to_tensor(Image.open(img1_path))
    img2 = to_tensor(Image.open(img2_path))
    quality_psnr = PSNR(img1, img2)
    return quality_psnr


def calculate_sequence_psnr(cur_dict, qp_list, layer: str = 'BL', gop_size=12, generate_recon=False):
    """
    Args:
        layer: BL or EL
        qp_list: qp setting
        gt_path: ground trurt rgb images' path
        rec_path: rec .yuv path

    Returns:
        several qp's corrosponding psnr quality
    """
    seq_name = cur_dict['seq_name']
    gt_path = cur_dict['gt_path']
    compressed_path = cur_dict['compressed_path']
    width_hr, height_hr = cur_dict['x1']['width'], cur_dict['x1']['height']
    ds_scale = 'x2'
    if 'x1_5' in cur_dict:
        widht_lr, height_lr = cur_dict['x1_5']['width'], cur_dict['x1_5']['height']
        ds_scale = 'x1_5'
    else:
        widht_lr, height_lr = cur_dict['x2']['width'], cur_dict['x2']['height']
    input_chroma_format = cur_dict['chroma']
    if layer == 'BL':
        width = widht_lr
        height = height_lr
        resolution = f'{width}x{height}'
    else:
        width = width_hr
        height = height_hr
        resolution = f'{width}x{height}'

    output_dir = os.path.join(compressed_path, seq_name, ds_scale)
    output_dir = path_transformer(output_dir)
    sequence_dict = {}
    for qp in tqdm(qp_list, desc='[PSNR] ', position=0):
        model_dict = {}
        rec_yuv_path = path_transformer(os.path.join(output_dir, f'{qp}_{layer}.yuv'))
        qp_dir_path = path_transformer(os.path.join(output_dir, f'{qp}_{layer}'))
        if generate_recon:
            if os.path.exists(qp_dir_path):
                shutil.rmtree(qp_dir_path)
            os.mkdir(qp_dir_path)

            yuv2png_command = f"Z:/user/bianyifan/ffmpeg/bin/ffmpeg.exe -pix_fmt {input_chroma_format}p -loglevel quiet -s {resolution} -i {rec_yuv_path} -f image2 {qp_dir_path}/im%05d.png"
            yuv2png_command = yuv2png_command
            os.system(yuv2png_command)
        # calculate psnr with gt
        rec_img_names = os.listdir(qp_dir_path)
        rec_img_names.sort()
        psnr_i_sum = 0
        psnr_p_sum = 0
        frame_num = len(rec_img_names)
        print(f'\n|Layer:{layer}|QP:{qp}| --> Frame num:{frame_num}')
        for index, img_name in enumerate(rec_img_names):
            if index % gop_size == 0 and layer == 'BL':
                img1_path = os.path.join(gt_path, seq_name, ds_scale, img_name)
                img2_path = os.path.join(qp_dir_path, img_name)
                psnr = psnr_between_pic(img1_path, img2_path)
                psnr_i_sum += psnr
            elif layer == 'BL':
                img1_path = os.path.join(gt_path, seq_name, ds_scale, img_name)
                img2_path = os.path.join(qp_dir_path, img_name)
                psnr = psnr_between_pic(img1_path, img2_path)
                psnr_p_sum += psnr
            elif index % gop_size == 0 and layer == 'EL':
                img1_path = os.path.join(gt_path, seq_name, 'x1', img_name)
                img2_path = os.path.join(qp_dir_path, img_name)
                psnr = psnr_between_pic(img1_path, img2_path)
                psnr_i_sum += psnr
            elif layer == 'EL':
                img1_path = os.path.join(gt_path, seq_name, 'x1', img_name)
                img2_path = os.path.join(qp_dir_path, img_name)
                psnr = psnr_between_pic(img1_path, img2_path)
                psnr_p_sum += psnr
        i_num = (frame_num - 1) // gop_size + 1
        p_num = frame_num - i_num
        psnr_i_avg = psnr_i_sum / i_num
        psnr_p_avg = psnr_p_sum / p_num
        psnr_a_avg = (psnr_i_sum + psnr_p_sum) / frame_num

        # write into dict
        model_dict['ave_i_frame_psnr'] = psnr_i_avg
        model_dict['ave_p_frame_psnr'] = psnr_p_avg
        model_dict['ave_all_frame_psnr'] = psnr_a_avg
        sequence_dict[f'{qp}.model'] = model_dict

        print(f'|Layer:{layer}|QP:{qp}| --> average all-psnr:{psnr_a_avg:.2f}')
    return sequence_dict


def calculate_sequence_psnr_hm(cur_dict, qp_list, layer: str = 'BL', gop_size=12, generate_recon=False):
    """
    Args:
        layer: BL or EL
        qp_list: qp setting
        gt_path: ground trurt rgb images' path
        rec_path: rec .yuv path

    Returns:
        several qp's corrosponding psnr quality
    """
    seq_name = cur_dict['seq_name']
    gt_path = cur_dict['gt_path']
    compressed_path = cur_dict['compressed_path']
    width_hr, height_hr = cur_dict['x1']['width'], cur_dict['x1']['height']
    input_chroma_format = cur_dict['chroma']
    width = width_hr
    height = height_hr
    resolution = f'{width}x{height}'

    output_dir = os.path.join(compressed_path, seq_name)
    output_dir = path_transformer(output_dir)
    sequence_dict = {}
    for qp in tqdm(qp_list, desc='[PSNR] ', position=0):
        model_dict = {}
        rec_yuv_path = path_transformer(os.path.join(output_dir, f'{qp}_{layer}.yuv'))
        qp_dir_path = path_transformer(os.path.join(output_dir, f'{qp}_{layer}'))
        if generate_recon:
            if os.path.exists(qp_dir_path):
                shutil.rmtree(qp_dir_path)
            os.mkdir(qp_dir_path)

            yuv2png_command = f"Z:/user/bianyifan/ffmpeg/bin/ffmpeg.exe -pix_fmt {input_chroma_format}p -s {resolution} -i {rec_yuv_path} -f image2 {qp_dir_path}/im%05d.png"
            yuv2png_command = yuv2png_command
            os.system(yuv2png_command)
            print(yuv2png_command)
            # -loglevel quiet
        # calculate psnr with gt
        rec_img_names = os.listdir(qp_dir_path)
        rec_img_names.sort()
        psnr_i_sum = 0
        psnr_p_sum = 0
        frame_num = len(rec_img_names)
        print(f'\n|Layer:{layer}|QP:{qp}| --> Frame num:{frame_num}')
        for index, img_name in enumerate(rec_img_names):
            if index % gop_size == 0 and layer == 'EL':
                img1_path = os.path.join(gt_path, seq_name, 'x1', img_name)
                img2_path = os.path.join(qp_dir_path, img_name)
                psnr = psnr_between_pic(img1_path, img2_path)
                psnr_i_sum += psnr
            elif layer == 'EL':
                img1_path = os.path.join(gt_path, seq_name, 'x1', img_name)
                img2_path = os.path.join(qp_dir_path, img_name)
                psnr = psnr_between_pic(img1_path, img2_path)
                psnr_p_sum += psnr
        i_num = (frame_num - 1) // gop_size + 1
        p_num = frame_num - i_num
        psnr_i_avg = psnr_i_sum / i_num
        psnr_p_avg = psnr_p_sum / p_num
        psnr_a_avg = (psnr_i_sum + psnr_p_sum) / frame_num

        # write into dict
        model_dict['ave_i_frame_psnr'] = psnr_i_avg
        model_dict['ave_p_frame_psnr'] = psnr_p_avg
        model_dict['ave_all_frame_psnr'] = psnr_a_avg
        sequence_dict[f'{qp}.model'] = model_dict

        print(f'|Layer:{layer}|QP:{qp}| --> average all-psnr:{psnr_a_avg:.2f}')
    return sequence_dict


def filter_dict(dict):
    not_keys = ['seq_name', 'ds_scale', 'ratio']
    res = {k: v for k, v in dict.items() if k not in not_keys}
    return res


def path_transformer(path_str, device = 'windows'):
    if device == 'windows':
        path_str = path_str.replace("\\\\","/").replace("\\","/")
    return path_str
