import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

codec_names = ['BPG', 'BPG444', 'VTM', 'iWave++', 'Factorized', 'Hyperprior', 'Autoregressive', 'Cheng2020', 'ELIC', 'MLIC++']
all_img_bpps = dict([(key,[]) for key in codec_names])
all_img_psnrs = dict([(key,[]) for key in codec_names])
all_img_msssims = dict([(key,[]) for key in codec_names])
fontdict = {'family' : 'Times New Roman', 'size'   : 16}

for img_idx in range(40):
    img_name = str(img_idx+1).zfill(2)
    
    ls = '-'
    marker = 'o'
    for codec_name in codec_names:
        with open(f'./{codec_name}.json') as f:
            json_dict = json.load(f)
            curr_bpps = json_dict[img_name]['bpp']
            curr_psnr = json_dict[img_name]['psnr']
            curr_msssim = json_dict[img_name]['ms_ssim']

        plt.figure(1)
        if codec_name != 'VTM':
            plt.plot(curr_bpps,curr_psnr,label=codec_name, marker=marker, ls=ls)
        else:
            plt.plot(curr_bpps, curr_psnr, label='H.266/VVC', marker=marker, ls=ls)

        plt.figure(2)
        if codec_name != 'VTM':
            plt.plot(curr_bpps,curr_msssim,label=codec_name, marker=marker, ls=ls)
        else:
            plt.plot(curr_bpps, curr_msssim, label='H.266/VVC', marker=marker, ls=ls)

        if len(all_img_bpps[codec_name]) == 0:
            all_img_bpps[codec_name] = np.array(curr_bpps) / 40
            all_img_psnrs[codec_name] = np.array(curr_psnr) / 40
            all_img_msssims[codec_name] = np.array(curr_msssim) / 40
        else:
            all_img_bpps[codec_name] += np.array(curr_bpps) / 40
            all_img_psnrs[codec_name] += np.array(curr_psnr) / 40
            all_img_msssims[codec_name] += np.array(curr_msssim) / 40
    
    fig = plt.figure(1)
    plt.legend(loc='best')
    plt.ylabel('PSNR (dB)', fontdict=fontdict)
    plt.xlabel('bits per pixel (bpp)', fontdict=fontdict)
    plt.grid()
    plt.subplots_adjust(left=0.13, right=0.995, top=0.990, bottom=0.105)
    axes = fig.get_axes()
    axes[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.savefig(f'./figs/PSNR_{img_name}.png')
    plt.clf()
    fig = plt.figure(2)
    plt.legend(loc='best')
    plt.ylabel('MS-SSIM', fontdict=fontdict)
    plt.xlabel('bits per pixel (bpp)', fontdict=fontdict)
    plt.grid()
    plt.subplots_adjust(left=0.105, right=0.995, top=0.990, bottom=0.105)
    axes = fig.get_axes()
    axes[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.savefig(f'./figs/MSSSIM_{img_name}.png')
    plt.cla()
    
ls = '-'
marker = 'o'
for codec_name in codec_names:
    if codec_name == 'BPG':
        all_img_bpps[codec_name] = all_img_bpps[codec_name][3:]
        all_img_psnrs[codec_name] = all_img_psnrs[codec_name][3:]
        all_img_msssims[codec_name] = all_img_msssims[codec_name][3:]
    elif codec_name == 'BPG444' or codec_name == 'iWave++' or codec_name == 'VTM':
        all_img_bpps[codec_name] = all_img_bpps[codec_name][1:]
        all_img_psnrs[codec_name] = all_img_psnrs[codec_name][1:]
        all_img_msssims[codec_name] = all_img_msssims[codec_name][1:]
    plt.figure(1)
    if codec_name != 'VTM':
        plt.plot(all_img_bpps[codec_name],all_img_psnrs[codec_name],label=codec_name, marker=marker, ls=ls)
    else:
        plt.plot(all_img_bpps[codec_name], all_img_psnrs[codec_name], label='H.266/VVC', marker=marker, ls=ls)
    plt.figure(2)
    if codec_name != 'VTM':
        plt.plot(all_img_bpps[codec_name],all_img_msssims[codec_name],label=codec_name, marker=marker, ls=ls)
    else:
        plt.plot(all_img_bpps[codec_name], all_img_msssims[codec_name], label='H.266/VVC', marker=marker, ls=ls)
    all_img_bpps[codec_name] = all_img_bpps[codec_name].tolist()
    all_img_psnrs[codec_name] = all_img_psnrs[codec_name].tolist()
    all_img_msssims[codec_name] = all_img_msssims[codec_name].tolist()

plt.figure(1)
plt.ylabel('PSNR (dB)', fontdict=fontdict)
plt.xlabel('bits per pixel (bpp)', fontdict=fontdict)
plt.grid()
plt.legend(loc='best', prop={'size': 11.5})
plt.subplots_adjust(left=0.105, right=0.995, top=0.990, bottom=0.105)
plt.savefig(f'./figs/PSNR_AVG.pdf')
plt.clf()

plt.figure(2)
plt.ylabel('MS-SSIM', fontdict=fontdict)
plt.xlabel('bits per pixel (bpp)', fontdict=fontdict)
plt.grid()
plt.legend(loc='best', prop={'size': 11.5})
plt.subplots_adjust(left=0.105, right=0.995, top=0.990, bottom=0.105)
plt.savefig(f'./figs/MSSSIM_AVG.pdf')
plt.cla()



            
    
