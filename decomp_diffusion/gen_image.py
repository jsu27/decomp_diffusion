import os
import numpy as np
import torch as th

from torchvision.utils import make_grid, save_image
from imageio import imread
from skimage.transform import resize as imresize

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
    if sample_method == 'ddim':
        model = gd._wrap_model(model)

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
    if sample_method == 'ddim':
        model = gd._wrap_model(model)

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


def gen_image_and_components_progressive(model, gd, steps=10, separate=False, num_components=4, sample_method='ddpm', im_path='clevr_im_10.png', batch_size=1, image_size=64, device='cuda', model_kwargs=None, desc='', save_dir='', dataset='clevr'):
    """Generate row of orig image, individual components, and reconstructed image"""
    sep = '_' if len(desc) > 0 else ''
    orig_img = get_im(im_path, resolution=image_size)
    latent = model.encode_latent(orig_img)
    model_kwargs = {'latent': latent}

    if separate:
        save_image(orig_img.cpu(), os.path.join(save_dir, f'{dataset}_{image_size}{sep}{desc}_orig.png')) # save indiv orig

    if len(save_dir) > 0:
        os.makedirs(save_dir, exist_ok=True)
    div = gd.num_timesteps // steps

    # generate imgs
    all_samples = [orig_img]

    # individual components
    for j in range(num_components):
        samples_progressive = []
        model_kwargs['latent_index'] = j
        if separate:
            os.makedirs(os.path.join(save_dir, f'comp_{j}_progressive/'), exist_ok=True)
        
        ind = 0
        for sample in gd.p_sample_loop_progressive(
            model,
            (batch_size, 3, image_size, image_size),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
        ):
            if ind % div == 0:
                samples_progressive.append(sample['sample'])
                if separate:
                    save_image(sample['sample'].cpu(), os.path.join(save_dir, f'comp_{j}_progressive/') + f'step_{ind}.png')
            ind += 1
        samples_progressive.append(sample['sample']) # add final result
        if separate:
            save_image(sample['sample'].cpu(), os.path.join(save_dir, f'comp_{j}_progressive/') + f'step_{ind}.png')
        
        # save progressive row
        samples = th.cat(samples_progressive, dim=0).cpu()   
        grid = make_grid(samples, nrow=samples.shape[0], padding=0)
        save_image(grid, os.path.join(save_dir, f'{dataset}_{image_size}{sep}{desc}_comp_{j}_progressive_row.png'))

        # save indiv comp
        if separate:
            save_image(sample['sample'].cpu(), os.path.join(save_dir, f'{dataset}_{image_size}{sep}{desc}_{j}.png'))
        all_samples.append(sample['sample'])
    
    # reconstruction
    model_kwargs['latent_index'] = None
    samples_progressive = []
    ind = 0
    if separate:
        os.makedirs(os.path.join(save_dir, 'composition_progressive/'), exist_ok=True)
    for sample in gd.p_sample_loop_progressive(
        model,
        (batch_size, 3, image_size, image_size),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
    ):
        if ind % div == 0:
            samples_progressive.append(sample['sample'])
            if separate:
                save_image(sample['sample'].cpu(), os.path.join(save_dir, 'composition_progressive/') + f'step_{ind}.png')
        ind += 1
    samples_progressive.append(sample['sample']) # add final result
    if separate:
        save_image(sample['sample'].cpu(), os.path.join(save_dir, 'composition_progressive/') + f'step_{ind}.png')

    # save progressive row
    samples = th.cat(samples_progressive, dim=0).cpu()   
    grid = make_grid(samples, nrow=samples.shape[0], padding=0)
    save_image(grid, os.path.join(save_dir, f'{dataset}_{image_size}{sep}{desc}_composition_progressive_row.png'))

    # save indiv reconstruction
    if separate:
        save_image(sample['sample'].cpu(), os.path.join(save_dir, f'{dataset}_{image_size}{sep}{desc}.png'))
    all_samples.append(sample['sample'])

    samples = th.cat(all_samples, dim=0).cpu()   
    grid = make_grid(samples, nrow=samples.shape[0], padding=0)
    # save row
    save_image(grid, os.path.join(save_dir, f'{dataset}_{image_size}{sep}{desc}_row.png'))


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
    
    gen_image_and_components(gen_model, gd, separate=separate, num_components=model.num_components, sample_method=sample_method, im_path=im_path, batch_size=batch_size, image_size=image_size, device=device, model_kwargs=model_kwargs, num_images=num_images, desc=desc, save_dir=save_dir, dataset=dataset)
    

def combine_components_slice(model, gd, indices=None, sample_method='ddim', im1='clevr_im_10.png', im2='clevr_im_25.png', device='cuda', num_images=4, model_kwargs={}, desc='', save_dir='', dataset='clevr', image_size=64):
    """Combine by adding components together
    """
    assert sample_method in ('ddpm', 'ddim')
    
    im1 = get_im(im_path=im1, resolution=image_size)
    im2 = get_im(im_path=im2, resolution=image_size)
    all_samples = [im1, im2]

    latent1 = model.encode_latent(im1)
    latent2 = model.encode_latent(im2)

    num_comps = model.num_components 

    # get latent slices
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
    model_kwargs['latent'] = combined_latent
    
    sample_loop_func = gd.p_sample_loop if sample_method == 'ddpm' else gd.ddim_sample_loop
    if sample_method == 'ddim':
        model = gd._wrap_model(model)

    # sampling loop
    sample = sample_loop_func(
            model,
            (1, 3, image_size, image_size),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:1]

    all_samples.append(sample)

    samples = th.cat(all_samples, dim=0).cpu()   
    grid = make_grid(samples, nrow=3, padding=0)
    save_image(grid, os.path.join(save_dir, f'{dataset}_{image_size}{desc}_row.png'))

def get_combined_model_add(model, latent1, latent2):
    def gen_model(x, ts, **model_kwargs):
        model_kwargs['latent'] = latent1
        out1 = model(x, ts, **model_kwargs)
        model_kwargs['latent'] = latent2
        out2 = model(x, ts, **model_kwargs)
        return (out1 + out2) / 2
    return gen_model


def combine_components_add(model, gd, sample_method='ddim', im1='clevr_im_10.png', im2='clevr_im_25.png', device='cuda', num_images=4, model_kwargs={}, desc='', save_dir='', dataset='clevr', image_size=64):
    """Combine by adding components together
    """
    assert sample_method in ('ddpm', 'ddim')

    im1 = get_im(im_path=im1, resolution=image_size)
    im2 = get_im(im_path=im2, resolution=image_size)
    all_samples = [im1, im2]

    latent1 = model.encode_latent(im1)
    latent2 = model.encode_latent(im2)

    gen_model = get_combined_model_add(model, latent1, latent2)
    
    sample_loop_func = gd.p_sample_loop if sample_method == 'ddpm' else gd.ddim_sample_loop
    if sample_method == 'ddim':
        gen_model = gd._wrap_model(gen_model)
    
    sample = sample_loop_func(
            gen_model,
            (1, 3, image_size, image_size),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:1]

    all_samples.append(sample)

    samples = th.cat(all_samples, dim=0).cpu()   
    grid = make_grid(samples, nrow=3, padding=0)
    save_image(grid, os.path.join(save_dir, f'{dataset}_{image_size}{desc}_row.png'))

def get_combined_model_cross(model1, model2, latent1, latent2, indices):
    num_comps = model1.num_components
    def gen_model(x_t, t, **model_kwargs):
        result = []
        for i in range(len(indices)):
            model_kwargs['latent_index'] = i
            target_model = model1 if indices[i] == 1 else model2
            target_latent = latent1 if indices[i] == 1 else latent2
            model_kwargs['latent'] = target_latent
            out = target_model(x_t, t, **model_kwargs)
            result.append(out) # b, 3, 64, 64
        result = th.cat(result, dim=0) # 4b, 3, 64, 64 
        b = x_t.shape[0]

        result = result.reshape(-1, num_comps, *x_t.shape[1:]) # b, 4, 3, 64, 64
        result = result.mean(dim=1) # b, 3, 64, 64
        return result

    return gen_model

def combine_components_cross_dataset(model, model2, gd, indices=None, sample_method='ddim', im1='im_19.jpg', im2='im_02.jpg', device='cuda', num_images=4, model_kwargs={}, desc='', save_dir='', dataset='celebahq', combine_method='add', image_size=64):
    """Combine across 2 models trained on different datasets
    """
    assert sample_method in ('ddpm', 'ddim')
    sample_loop_func = gd.p_sample_loop if sample_method == 'ddpm' else gd.ddim_sample_loop

    im1 = get_im(im_path=im1, resolution=image_size)
    im2 = get_im(im_path=im2, resolution=image_size)
    all_samples = [im1.cpu(), im2.cpu()]

    latent1 = model.encode_latent(im1)
    latent2 = model2.encode_latent(im2)

    latent_dim = model.latent_dim
    num_comps = model.num_components 

    # get latent slices
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
    
    gen_model = get_combined_model_cross(model, model2, latent1, latent2, indices)
    if sample_method == 'ddim':
        gen_model = gd._wrap_model(gen_model)

    samples = []
    sample = sample_loop_func(
            gen_model,
            (1, 3, image_size, image_size),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:1]
        
    all_samples.append(sample.cpu())

    samples = th.cat(all_samples, dim=0)
    grid = make_grid(samples, nrow=3, padding=0)
    save_image(grid, os.path.join(save_dir, f'{dataset}_{image_size}{desc}_row.png'))

