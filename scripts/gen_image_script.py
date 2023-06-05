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
from ema_pytorch import EMA
from decomp_diffusion.image_datasets import get_dataset
from decomp_diffusion.model_and_diffusion_util import *
from decomp_diffusion.diffusion.respace import SpacedDiffusion, _WrappedModel
from decomp_diffusion.gen_image import *

# fix randomness
th.manual_seed(0)
np.random.seed(0)


# def get_im(im_path='im_10_clevr.png', resolution=64):
#     im = imread(im_path)
#     if 'kitti' in im_path: # kitti, vkitti
#         im = im[:, 433:808, :]
#     im = imresize(im, (resolution, resolution))[:, :, :3]
#     im = th.Tensor(im).permute(2, 0, 1)[None, :, :, :].contiguous().cuda()
#     return im


# def gen_image(model, gd, sample_method='ddim', batch_size=1, image_size=64, device='cuda', model_kwargs=None, num_images=4, desc='', save_dir='', dataset='clevr'):
#     all_samples = []
#     sample_loop_func = gd.p_sample_loop if sample_method == 'ddpm' else gd.ddim_sample_loop
#     # generate imgs
#     for i in range(num_images):
#         samples = sample_loop_func(
#             model,
#             (batch_size, 3, image_size, image_size),
#             device=device,
#             clip_denoised=True,
#             progress=True,
#             model_kwargs=model_kwargs,
#             cond_fn=None,
#         )[:batch_size]

#         all_samples.append(samples)

#     samples = th.cat(all_samples, dim=0).cpu()   
#     grid = make_grid(samples, nrow=int(samples.shape[0] ** 0.5), padding=0)
#     if len(desc) > 0:
#         desc = '_' + desc
#     if len(save_dir) > 0:
#         os.makedirs(save_dir, exist_ok=True)

#     save_image(grid, os.path.join(save_dir, f'{dataset}_{image_size}{desc}.png'))


# def gen_image_and_components(model, gd, separate=False, num_components=4, sample_method='ddim', im_path='im_10.png', batch_size=1, image_size=64, device='cuda', model_kwargs=None, num_images=4, desc='', save_dir='', dataset='clevr'):
#     """Generate row of orig image, individual components, and reconstructed image"""
#     sep = '_' if len(desc) > 0 else ''
#     orig_img = get_im(im_path, resolution=image_size)
#     if separate:
#         save_image(orig_img.cpu(), os.path.join(save_dir, f'{dataset}_{image_size}{sep}{desc}_orig.png')) # save indiv orig

#     if len(save_dir) > 0:
#         os.makedirs(save_dir, exist_ok=True)
#     assert sample_method in ('ddpm', 'ddim')
#     sample_loop_func = gd.p_sample_loop if sample_method == 'ddpm' else gd.ddim_sample_loop

#     # generate imgs
#     for i in range(num_images):
#         all_samples = [orig_img]
#         # individual components
#         for j in range(num_components):
#             model_kwargs['latent_index'] = j
#             sample = sample_loop_func(
#                 model,
#                 (batch_size, 3, image_size, image_size),
#                 device=device,
#                 clip_denoised=True,
#                 progress=True,
#                 model_kwargs=model_kwargs,
#                 cond_fn=None,
#             )[:batch_size]

#             # save indiv comp
#             if separate:
#                 save_image(sample.cpu(), os.path.join(save_dir, f'{dataset}_{image_size}{sep}{desc}_{i}_{j}.png'))
#             all_samples.append(sample)
#         # reconstruction
#         model_kwargs['latent_index'] = None
#         sample = sample_loop_func(
#             model,
#             (batch_size, 3, image_size, image_size),
#             device=device,
#             clip_denoised=True,
#             progress=True,
#             model_kwargs=model_kwargs,
#             cond_fn=None,
#         )[:batch_size]
#         # save indiv reconstruction
#         if separate:
#             save_image(sample.cpu(), os.path.join(save_dir, f'{dataset}_{image_size}{sep}{desc}_{i}.png'))
#         all_samples.append(sample)

#         samples = th.cat(all_samples, dim=0).cpu()   
#         grid = make_grid(samples, nrow=samples.shape[0], padding=0)
#         # save row
#         save_image(grid, os.path.join(save_dir, f'{dataset}_{image_size}{sep}{desc}_row{i}.png'))

# # classifier-free guidance
# def get_model_fn(model, gd, batch_size=1, guidance_scale=10.0, device='cuda'):
#     zeros_latent = th.zeros(batch_size, model.latent_dim_expand).to(device)

#     def model_fn(x_t, ts, **model_kwargs):
#         x_start = model(x_t, ts, **model_kwargs) # forward(self, x, t, x_start=None, latent=None, latent_index=None):
#         eps = gd._predict_eps_from_xstart(x_t, ts, x_start) #  x_t, t, pred_xstart

#         zeros_kwargs = model_kwargs.copy()
#         zeros_kwargs['latent'] = zeros_latent # pass in 0s
    
#         uncond_x_start = model(x_t, ts, **zeros_kwargs)
#         uncond_eps = gd._predict_eps_from_xstart(x_t, ts, uncond_x_start)

#         guided_eps = uncond_eps + guidance_scale * (eps - uncond_eps)
#         result = gd._predict_xstart_from_eps(x_t, ts, guided_eps) #  x_t, t, eps)
#         return result

#     return model_fn


# def get_gen_images(model, gd, sample_method='ddim', im_path='im_10.png', latent=None, batch_size=1, image_size=64, device='cuda', model_kwargs=None, num_images=4, desc='', save_dir='', free=False, guidance_scale=10.0, dataset='clevr', separate=False):
#     orig_im = get_im(im_path=im_path, resolution=image_size)
#     if latent == None:
#         latent = model.encode_latent(orig_im)
#         model_kwargs = {'latent': latent}

#     if device == None:
#         device = model.dev()

#     # use classifier free
#     gen_model = model if not free else get_model_fn(model, gd, batch_size=batch_size, guidance_scale=guidance_scale)
    
#     # if not separate:
#     gen_image_and_components(gen_model, gd, separate=separate, num_components=model.num_components, sample_method=sample_method, im_path=im_path, batch_size=batch_size, image_size=image_size, device=device, model_kwargs=model_kwargs, num_images=num_images, desc=desc, save_dir=save_dir, dataset=dataset)
    
#     # else:
#     #     basename = os.path.basename(im_path)
#     #     save_image(orig_im, os.path.join(save_dir, f'orig_{basename}')) # save original

#     #     gen_image(gen_model, gd, batch_size=batch_size, image_size=image_size, device=device, model_kwargs=model_kwargs, num_images=num_images, desc=desc, save_dir=save_dir, dataset=dataset)

#     #     # sample with 1 latent at a time
#     #     num_comps = model.num_components
#     #     latent_dim = latent.shape[1] // num_comps # length of single latent
#     #     for i in range(num_comps):
#     #         model_kwargs['latent_index'] = i
#     #         gen_image(gen_model, gd, batch_size=batch_size, image_size=image_size, device=device, model_kwargs=model_kwargs, num_images=num_images, desc=desc+'_'+str(i), save_dir=save_dir, dataset=dataset)

# def combine_components(model, gd, indices=None, sample_method='ddim', im1='im_19.jpg', im2='im_02.jpg', device='cuda', num_images=4, model_kwargs={}, desc='', save_dir='', dataset='celebahq', combine_method='add', image_size=64):
#     """Combine by adding components together
#     combine_method: 'add', 'slice'
#     """
#     assert combine_method in ('add', 'slice')
#     assert sample_method in ('ddpm', 'ddim')
#     sample_loop_func = gd.p_sample_loop if sample_method == 'ddpm' else gd.ddim_sample_loop

#     im1 = get_im(im_path=im1, resolution=image_size)
#     im2 = get_im(im_path=im2, resolution=image_size)
#     all_samples = [im1, im2]

#     latent1 = model.encode_latent(im1)
#     latent2 = model.encode_latent(im2)

#     latent_dim = model.latent_dim
#     num_comps = model.num_components 
    
#     if combine_method == 'add':
#         combined_latent = (latent1 + latent2) / 2 # averaged
#     else:
#         if indices == None:
#             half = num_comps // 2
#             indices = [1] * half + [0] * half # first half 1, second half 0
#             indices = th.Tensor(indices) == 1
#             indices = indices.reshape(num_comps, 1)
#         elif type(indices) == str:
#             indices = indices.split(',')
#             indices = [int(ind) for ind in indices]
#             indices = th.Tensor(indices).reshape(-1, 1) == 1
#         assert len(indices) == num_comps
#         indices = indices.to(device)

#         latent1 = latent1.reshape(num_comps, -1).to(device)
#         latent2 = latent2.reshape(num_comps, -1).to(device)

#         combined_latent = th.where(indices, latent1, latent2)
#         combined_latent = combined_latent.reshape(1, -1)
#         # combined_latent = th.cat((latent1[:, :latent_dim * idx], latent2[:, latent_dim * idx:]), axis=1)
#     model_kwargs['latent'] = combined_latent
    
#     # gen_image(model, gd, sample_method=sample_method, device=device, model_kwargs=model_kwargs, num_images=num_images, desc=desc, save_dir=save_dir, dataset=dataset)
#     sample = sample_loop_func(
#             model,
#             (1, 3, image_size, image_size),
#             device=device,
#             clip_denoised=True,
#             progress=True,
#             model_kwargs=model_kwargs,
#             cond_fn=None,
#         )[:1]

#     save_image(sample, os.path.join(save_dir, f'{dataset}_{image_size}{desc}_combo.png')) # same combo img separately

#     all_samples.append(sample)

#     samples = th.cat(all_samples, dim=0).cpu()   
#     grid = make_grid(samples, nrow=3, padding=0)
#     if len(desc) > 0:
#         desc = '_' + desc
#     if len(save_dir) > 0:
#         os.makedirs(save_dir, exist_ok=True)

#     save_image(grid, os.path.join(save_dir, f'{dataset}_{image_size}{desc}_row.png'))

if __name__=='__main__':
    """
    Generate images. Several use cases:
        - Generate single image: provide --im_path
        - Generate multiple images from a dataset: provide --gen_images
        - Generate combinations from a dataset: provide --combine
        - Generate image components in separate files: provide --separate
    """
    defaults = model_defaults()
    defaults.update(unet_model_defaults())
    defaults.update(diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--free', action='store_true') # if arg not provided, False
    parser.add_argument('--guidance_scale', type=float, default=10.0)
    parser.add_argument('--im_path', default='im_10_clevr.png')
    # parser.add_argument('--combine', action='store_true')
    parser.add_argument('--dataset', default='clevr')
    parser.add_argument('--dataset2', default=None) # for multi-modal combination
    parser.add_argument('--gen_images', default=0, type=int) # generate many images from dataset
    parser.add_argument('--sample_method', default='ddim')
    parser.add_argument('--separate', action='store_true')
    parser.add_argument('--num_images', default=4, type=int) # if gen 1 img, how many samples
    parser.add_argument('--combine_method', default=None)
    parser.add_argument('--indices', type=str, default=None) # comma-delimited list
    parser.add_argument('--clevr_combine', action='store_true') # hard coded case
    parser.add_argument('--clevr_indiv', action='store_true')
    args = parser.parse_args()

    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')

    ckpt_path = args.ckpt_path
    save_dir = args.save_dir
    if len(save_dir) > 0:
        os.makedirs(save_dir, exist_ok=True)
    guidance_scale = args.guidance_scale
    free = args.free
    dataset = args.dataset
    dataset2 = args.dataset2
    if dataset2 == None:
        dataset2 = dataset
    image_size = args.image_size
    sample_method = args.sample_method
    separate = args.separate
    combine_method = args.combine_method
    indices = args.indices

    model_desc = args.model_desc
    training_model_defaults = unet_model_defaults() if model_desc == 'unet_model' else model_defaults()
    model_kwargs = args_to_dict(args, training_model_defaults.keys())
    model = create_diffusion_model(**model_kwargs)
    if 'ema' in ckpt_path: # need EMA wrapper
        model = EMA(model, beta=0.99, update_every=1).unet_model
    model.eval()

    diffusion_kwargs = args_to_dict(args, diffusion_defaults().keys())
    gd = create_gaussian_diffusion(**diffusion_kwargs)

    num_images = args.num_images        

    # if has_cuda:
    #     model.convert_to_fp16()
    model.to(device)

    print(f'loading from {ckpt_path}')
    checkpoint = th.load(ckpt_path, map_location='cpu')

    model.load_state_dict(checkpoint)
    print(f'saving to {save_dir}')
    gen_images = args.gen_images
    step_size = 100 if 'kitti' in dataset or 'falcor' in dataset else 1 # kitti, vkitti have very similar adjacent imgs
    if sample_method == 'ddim':
        # use respaced diffusion steps
        desired_timesteps = 50 
        num_timesteps = diffusion_kwargs['steps']

        spacing = num_timesteps // desired_timesteps
        spaced_ts = list(range(0, num_timesteps + 1, spacing))
        betas = get_named_beta_schedule(diffusion_kwargs['noise_schedule'], num_timesteps)
        diffusion_kwargs['betas'] = betas
        del diffusion_kwargs['steps'], diffusion_kwargs['noise_schedule']
        gd = SpacedDiffusion(spaced_ts, rescale_timesteps=True, original_num_steps=num_timesteps, **diffusion_kwargs)
        
        model = gd._wrap_model(model)

        num_images = 1 # deterministic

    if gen_images > 0: # generate multiple imgs
        num_digits = len(str(gen_images * step_size))
        data = get_dataset(dataset, num_images=gen_images * step_size, resolution=image_size).images
        data2 = get_dataset(dataset2, num_images=gen_images * step_size, resolution=image_size).images
        if combine_method is None:
            for _i in range(gen_images):
                i = _i * step_size
                im = data[i]
                get_gen_images(model, gd, sample_method=sample_method, im_path=im, image_size=image_size, device=device, save_dir=save_dir, guidance_scale=guidance_scale, free=free, dataset=dataset, num_images=1, desc=f'im_{i:0{num_digits}}', separate=separate)
        else:
            for _i in range(gen_images):
                i = _i * step_size
                print('_i', _i)
                # save row for reference
                get_gen_images(model, gd, separate=False, sample_method=sample_method, im_path=data[i], image_size=image_size, device=device, save_dir=save_dir, guidance_scale=guidance_scale, free=free, dataset=dataset, num_images=num_images, desc=f'im_{i:0{num_digits}}')
                for _j in range(_i + 1, gen_images):
                    j = _j * step_size
                    if i == 0: # only save once on first pass
                        get_gen_images(model, gd, separate=False, sample_method=sample_method, im_path=data2[j], image_size=image_size, device=device, save_dir=save_dir, guidance_scale=guidance_scale, free=free, dataset=dataset2, num_images=num_images, desc=f'im_{j:0{num_digits}}')
                    combine_components(model, gd, image_size=image_size, indices=indices, im1=data[i], im2=data2[j], combine_method=combine_method, num_images=1, desc=f'comb{i}x{j}', save_dir=save_dir, dataset=dataset)

    elif combine_method is None:
        get_gen_images(model, gd, sample_method=sample_method, im_path=args.im_path, image_size=image_size, device=device, save_dir=save_dir, guidance_scale=guidance_scale, free=free, dataset=dataset, num_images=num_images, separate=separate)
    else:
        combine_components(model, gd, image_size=image_size, indices=indices, combine_method=combine_method, sample_method=sample_method, save_dir=save_dir, dataset=dataset, num_images=num_images)

        
    