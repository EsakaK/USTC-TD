from typing import Tuple, Union

import os
import platform
from tqdm import tqdm
import shutil
from PIL import Image
from torchvision import transforms
import scipy.ndimage

from torch import Tensor

YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}


def rgb_to_ycbcr420(rgb):
    '''
    input is 3xhxw RGB float numpy array, in the range of [0, 1]
    output is y: 1xhxw, uv: 2x(h/2)x(w/x), in the range of [0, 1]
    '''
    c, h, w = rgb.shape
    assert c == 3
    assert h % 2 == 0
    assert w % 2 == 0
    r, g, b = np.split(rgb, 3, axis=0)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5

    # to 420
    cb = np.mean(np.reshape(cb, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    cr = np.mean(np.reshape(cr, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    uv = np.concatenate((cb, cr), axis=0)

    y = np.clip(y, 0., 1.)
    uv = np.clip(uv, 0., 1.)

    return y, uv


def ycbcr420_to_rgb(y, uv, order=1):
    '''
    y is 1xhxw Y float numpy array, in the range of [0, 1]
    uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
    order: 0 nearest neighbor, 1: binear (default)
    return value is 3xhxw RGB float numpy array, in the range of [0, 1]
    '''
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    cb = uv[0:1, :, :]
    cr = uv[1:2, :, :]
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = np.concatenate((r, g, b), axis=0)
    rgb = np.clip(rgb, 0., 1.)
    return rgb


def ycbcr420_to_444(y, uv, order=1):
    '''
    y is 1xhxw Y float numpy array, in the range of [0, 1]
    uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
    order: 0 nearest neighbor, 1: binear (default)
    return value is 3xhxw YCbCr float numpy array, in the range of [0, 1]
    '''
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    yuv = np.concatenate((y, uv), axis=0)
    return yuv


def ycbcr444_to_420(yuv):
    '''
    input is 3xhxw YUV float numpy array, in the range of [0, 1]
    output is y: 1xhxw, uv: 2x(h/2)x(w/x), in the range of [0, 1]
    '''
    c, h, w = yuv.shape
    assert c == 3
    assert h % 2 == 0
    assert w % 2 == 0
    y, u, v = np.split(yuv, 3, axis=0)

    # to 420
    u = np.mean(np.reshape(u, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    v = np.mean(np.reshape(v, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    uv = np.concatenate((u, v), axis=0)

    y = np.clip(y, 0., 1.)
    uv = np.clip(uv, 0., 1.)

    return y, uv


def rgb_to_ycbcr(rgb):
    '''
    input is 3xhxw RGB float numpy array, in the range of [0, 1]
    output is yuv: 3xhxw, in the range of [0, 1]
    '''
    c, h, w = rgb.shape
    assert c == 3
    r, g, b = np.split(rgb, 3, axis=0)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5

    yuv = np.concatenate((y, cb, cr), axis=0)
    yuv = np.clip(yuv, 0., 1.)

    return yuv


def ycbcr_to_rgb(yuv):
    '''
    yuv is 3xhxw YCbCr float numpy array, in the range of [0, 1]
    return value is 3xhxw RGB float numpy array, in the range of [0, 1]
    '''
    y, cb, cr = np.split(yuv, 3, axis=0)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = np.concatenate((r, g, b), axis=0)
    rgb = np.clip(rgb, 0., 1.)
    return rgb


def _check_input_tensor(tensor: Tensor) -> None:
    if (
        not isinstance(tensor, Tensor)
        or not tensor.is_floating_point()
        or not len(tensor.size()) in (3, 4)
        or not tensor.size(-3) == 3
    ):
        raise ValueError(
            "Expected a 3D or 4D tensor with shape (Nx3xHxW) or (3xHxW) as input"
        )


def rgb2ycbcr(rgb: Tensor) -> Tensor:
    """RGB to YCbCr conversion for torch Tensor.
    Using ITU-R BT.709 coefficients.

    Args:
        rgb (torch.Tensor): 3D or 4D floating point RGB tensor

    Returns:
        ycbcr (torch.Tensor): converted tensor
    """
    _check_input_tensor(rgb)

    r, g, b = rgb.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    ycbcr = torch.cat((y, cb, cr), dim=-3)
    return ycbcr


def ycbcr2rgb(ycbcr: Tensor) -> Tensor:
    """YCbCr to RGB conversion for torch Tensor.
    Using ITU-R BT.709 coefficients.

    Args:
        ycbcr (torch.Tensor): 3D or 4D floating point RGB tensor

    Returns:
        rgb (torch.Tensor): converted tensor
    """
    #_check_input_tensor(ycbcr)
    #print(ycbcr)
    y, cb, cr = ycbcr.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = torch.cat((r, g, b), dim=-3)
    return rgb


def yuv_444_to_420(
    yuv: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
    mode: str = "avg_pool",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert a 444 tensor to a 420 representation.

    Args:
        yuv (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)): 444
            input to be downsampled. Takes either a (Nx3xHxW) tensor or a tuple
            of 3 (Nx1xHxW) tensors.
        mode (str): algorithm used for downsampling: ``'avg_pool'``. Default
            ``'avg_pool'``

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): Converted 420
    """
    if mode not in ("avg_pool",):
        raise ValueError(f'Invalid downsampling mode "{mode}".')

    if mode == "avg_pool":

        def _downsample(tensor):
            return F.avg_pool2d(tensor, kernel_size=2, stride=2)

    if isinstance(yuv, torch.Tensor):
        y, u, v = yuv.chunk(3, 1)
    else:
        y, u, v = yuv

    return (y, _downsample(u), _downsample(v))


def yuv_420_to_444(
    yuv: Tuple[Tensor, Tensor, Tensor],
    mode: str = "bilinear",
    return_tuple: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """Convert a 420 input to a 444 representation.

    Args:
        yuv (torch.Tensor, torch.Tensor, torch.Tensor): 420 input frames in
            (Nx1xHxW) format
        mode (str): algorithm used for upsampling: ``'bilinear'`` |
            ``'nearest'`` Default ``'bilinear'``
        return_tuple (bool): return input as tuple of tensors instead of a
            concatenated tensor, 3 (Nx1xHxW) tensors instead of one (Nx3xHxW)
            tensor (default: False)

    Returns:
        (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)): Converted
            444
    """
    if len(yuv) != 3 or any(not isinstance(c, torch.Tensor) for c in yuv):
        raise ValueError("Expected a tuple of 3 torch tensors")

    if mode not in ("bilinear", "nearest"):
        raise ValueError(f'Invalid upsampling mode "{mode}".')

    if mode in ("bilinear", "nearest"):

        def _upsample(tensor):
            return F.interpolate(tensor, scale_factor=2, mode=mode, align_corners=False)

    y, u, v = yuv
    u, v = _upsample(u), _upsample(v)
    if return_tuple:
        return y, u, v
    return torch.cat((y, u, v), dim=1)

def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def mse2PSNR(mse, data_range=1):
    if mse > 1e-10:
        psnr = 10 * np.log10(data_range * data_range / mse)
    else:
        psnr = 999.9
    return psnr


def PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def calc_ssim(img1, img2, data_range=255):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2

    return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                         (sigma1_sq + sigma2_sq + C2)),
            (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))

def calc_msssim(img1, img2, data_range=255):
    '''
    img1 and img2 are 2D arrays
    '''
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    height, width = img1.shape
    if height < 176 or width < 176:
        # according to HM implementation
        level = 4
        weight = np.array([0.0517, 0.3295, 0.3462, 0.2726])
    if height < 88 or width < 88:
        level = 3
        # according to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9308879/
        weight = np.array([0.2, 0.5, 0.3])
    downsample_filter = np.ones((2, 2)) / 4.0
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(level):
        ssim_map, cs_map = calc_ssim(im1, im2, data_range=data_range)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter,
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter,
                                                mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level - 1] ** weight[0:level - 1]) *
            (mssim[level - 1] ** weight[level - 1]))


def calc_msssim_rgb(img1, img2, data_range=255):
    '''
    img1 and img2 are arrays with 3xHxW
    '''
    msssim = 0
    for i in range(3):
        msssim += calc_msssim(img1[i, :, :], img2[i, :, :], data_range)
    return msssim / 3


#----------------------------------------------------------------------------------#
from .core import imresize
from torch import cuda
from .video_reader import YUVReader
from .video_writer import YUVWriter
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from scipy import signal
from scipy import ndimage

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
