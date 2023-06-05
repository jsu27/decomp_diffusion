import os
import numpy as np
import argparse
import torch as th

from torchvision.utils import make_grid, save_image
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from imageio import imread
from skimage.transform import resize as imresize
# from ema_pytorch import EMA

# from decomp_diffusion.image_datasets import get_dataset
# from util.model_and_diffusion_util import * # TODO refine
# from diffusion.respace import SpacedDiffusion #, _WrappedModel

# fix randomness
th.manual_seed(0)
np.random.seed(0)


def get_im(im_path='clevr_im_10.png', resolution=64):
    im = imread(im_path)
    if 'kitti' in im_path: # kitti, vkitti
        im = im[:, 433:808, :]
    im = imresize(im, (resolution, resolution))[:, :, :3]
    im = th.Tensor(im).permute(2, 0, 1)[None, :, :, :].contiguous().cuda()
    return im


def gen_image(model, gd, sample_method='ddim', batch_size=1, image_size=64, device='cuda', model_kwargs=None, num_images=4, desc='', save_dir='', dataset='clevr'):
    all_samples = []
    sample_loop_func = gd.p_sample_loop if sample_method == 'ddpm' else gd.ddim_sample_loop
    # generate imgs
    for i in range(num_images):
        samples = sample_loop_func(
            model,
            (batch_size, 3, image_size, image_size),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]

        all_samples.append(samples)

    samples = th.cat(all_samples, dim=0).cpu()   
    grid = make_grid(samples, nrow=int(samples.shape[0] ** 0.5), padding=0)
    if len(desc) > 0:
        desc = '_' + desc
    if len(save_dir) > 0:
        os.makedirs(save_dir, exist_ok=True)

    save_image(grid, os.path.join(save_dir, f'{dataset}_{image_size}{desc}.png'))


def gen_image_and_components(model, gd, separate=False, num_components=4, sample_method='ddim', im_path='clevr_im_10.png', batch_size=1, image_size=64, device='cuda', model_kwargs=None, num_images=4, desc='', save_dir='', dataset='clevr'):
    """Generate row of orig image, individual components, and reconstructed image"""
    sep = '_' if len(desc) > 0 else ''
    orig_img = get_im(im_path, resolution=image_size)
    if separate:
        save_image(orig_img.cpu(), os.path.join(save_dir, f'{dataset}_{image_size}{sep}{desc}_orig.png')) # save indiv orig

    if len(save_dir) > 0:
        os.makedirs(save_dir, exist_ok=True)
    assert sample_method in ('ddpm', 'ddim')
    sample_loop_func = gd.p_sample_loop if sample_method == 'ddpm' else gd.ddim_sample_loop

    # generate imgs
    for i in range(num_images):
        all_samples = [orig_img]
        # individual components
        for j in range(num_components):
            model_kwargs['latent_index'] = j
            sample = sample_loop_func(
                model,
                (batch_size, 3, image_size, image_size),
                device=device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                cond_fn=None,
            )[:batch_size]

            # save indiv comp
            if separate:
                save_image(sample.cpu(), os.path.join(save_dir, f'{dataset}_{image_size}{sep}{desc}_{i}_{j}.png'))
            all_samples.append(sample)
        # reconstruction
        model_kwargs['latent_index'] = None
        sample = sample_loop_func(
            model,
            (batch_size, 3, image_size, image_size),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        # save indiv reconstruction
        if separate:
            save_image(sample.cpu(), os.path.join(save_dir, f'{dataset}_{image_size}{sep}{desc}_{i}.png'))
        all_samples.append(sample)

        samples = th.cat(all_samples, dim=0).cpu()   
        grid = make_grid(samples, nrow=samples.shape[0], padding=0)
        # save row
        save_image(grid, os.path.join(save_dir, f'{dataset}_{image_size}{sep}{desc}_row{i}.png'))

# classifier-free guidance
def get_model_fn(model, gd, batch_size=1, guidance_scale=10.0, device='cuda'):
    zeros_latent = th.zeros(batch_size, model.latent_dim_expand).to(device)

    def model_fn(x_t, ts, **model_kwargs):
        x_start = model(x_t, ts, **model_kwargs) # forward(self, x, t, x_start=None, latent=None, latent_index=None):
        eps = gd._predict_eps_from_xstart(x_t, ts, x_start) #  x_t, t, pred_xstart

        zeros_kwargs = model_kwargs.copy()
        zeros_kwargs['latent'] = zeros_latent # pass in 0s
    
        uncond_x_start = model(x_t, ts, **zeros_kwargs)
        uncond_eps = gd._predict_eps_from_xstart(x_t, ts, uncond_x_start)

        guided_eps = uncond_eps + guidance_scale * (eps - uncond_eps)
        result = gd._predict_xstart_from_eps(x_t, ts, guided_eps) #  x_t, t, eps)
        return result

    return model_fn


def get_gen_images(model, gd, sample_method='ddim', im_path='clevr_im_10.png', latent=None, batch_size=1, image_size=64, device='cuda', model_kwargs=None, num_images=4, desc='', save_dir='', free=False, guidance_scale=10.0, dataset='clevr', separate=False):
    orig_im = get_im(im_path=im_path, resolution=image_size)
    if latent == None:
        latent = model.encode_latent(orig_im)
        model_kwargs = {'latent': latent}

    if device == None:
        device = model.dev()

    # use classifier free
    gen_model = model if not free else get_model_fn(model, gd, batch_size=batch_size, guidance_scale=guidance_scale)
    
    # if not separate:
    gen_image_and_components(gen_model, gd, separate=separate, num_components=model.num_components, sample_method=sample_method, im_path=im_path, batch_size=batch_size, image_size=image_size, device=device, model_kwargs=model_kwargs, num_images=num_images, desc=desc, save_dir=save_dir, dataset=dataset)
    
    # else:
    #     basename = os.path.basename(im_path)
    #     save_image(orig_im, os.path.join(save_dir, f'orig_{basename}')) # save original

    #     gen_image(gen_model, gd, batch_size=batch_size, image_size=image_size, device=device, model_kwargs=model_kwargs, num_images=num_images, desc=desc, save_dir=save_dir, dataset=dataset)

    #     # sample with 1 latent at a time
    #     num_comps = model.num_components
    #     latent_dim = latent.shape[1] // num_comps # length of single latent
    #     for i in range(num_comps):
    #         model_kwargs['latent_index'] = i
    #         gen_image(gen_model, gd, batch_size=batch_size, image_size=image_size, device=device, model_kwargs=model_kwargs, num_images=num_images, desc=desc+'_'+str(i), save_dir=save_dir, dataset=dataset)

def combine_components(model, gd, indices=None, sample_method='ddim', im1='im_19.jpg', im2='im_02.jpg', device='cuda', num_images=4, model_kwargs={}, desc='', save_dir='', dataset='celebahq', combine_method='add', image_size=64):
    """Combine by adding components together
    combine_method: 'add', 'slice'
    """
    assert combine_method in ('add', 'slice')
    assert sample_method in ('ddpm', 'ddim')
    sample_loop_func = gd.p_sample_loop if sample_method == 'ddpm' else gd.ddim_sample_loop

    im1 = get_im(im_path=im1, resolution=image_size)
    im2 = get_im(im_path=im2, resolution=image_size)
    all_samples = [im1, im2]

    latent1 = model.encode_latent(im1)
    latent2 = model.encode_latent(im2)

    latent_dim = model.latent_dim
    num_comps = model.num_components 
    
    if combine_method == 'add':
        combined_latent = (latent1 + latent2) / 2 # averaged
    else:
        if indices == None:
            half = num_comps // 2
            indices = [1] * half + [0] * half # first half 1, second half 0
            indices = th.Tensor(indices) == 1
            indices = indices.reshape(num_comps, 1)
        elif type(indices) == str:
            indices = indices.split(',')
            indices = [int(ind) for ind in indices]
            indices = th.Tensor(indices).reshape(-1, 1) == 1
        assert len(indices) == num_comps
        indices = indices.to(device)

        latent1 = latent1.reshape(num_comps, -1).to(device)
        latent2 = latent2.reshape(num_comps, -1).to(device)

        combined_latent = th.where(indices, latent1, latent2)
        combined_latent = combined_latent.reshape(1, -1)
        # combined_latent = th.cat((latent1[:, :latent_dim * idx], latent2[:, latent_dim * idx:]), axis=1)
    model_kwargs['latent'] = combined_latent
    
    # gen_image(model, gd, sample_method=sample_method, device=device, model_kwargs=model_kwargs, num_images=num_images, desc=desc, save_dir=save_dir, dataset=dataset)
    sample = sample_loop_func(
            model,
            (1, 3, image_size, image_size),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:1]

    save_image(sample, os.path.join(save_dir, f'{dataset}_{image_size}{desc}_combo.png')) # same combo img separately

    all_samples.append(sample)

    samples = th.cat(all_samples, dim=0).cpu()   
    grid = make_grid(samples, nrow=3, padding=0)
    if len(desc) > 0:
        desc = '_' + desc
    if len(save_dir) > 0:
        os.makedirs(save_dir, exist_ok=True)

    save_image(grid, os.path.join(save_dir, f'{dataset}_{image_size}{desc}_row.png'))
