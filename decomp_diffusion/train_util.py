import torch as th
import argparse
import os
import numpy as np
import copy
from ema_pytorch import EMA

from .model_and_diffusion_util import *
from .gen_image import get_gen_images

def uniform_sample_timesteps(steps, batch_size):
    indices_np = np.random.choice(steps, size=(batch_size,))
    indices = th.from_numpy(indices_np).long() # .to(device)
    return indices

def update_ema(target_params, source_params, ema_rate=0.99):
    # update target in-place
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(ema_rate).add_(src, alpha=1 - ema_rate)

def params_to_state_dict(target_params, model):
    state_dict = model.state_dict()
    for i, (name, _value) in enumerate(model.named_parameters()):
        assert name in state_dict
        state_dict[name] = target_params[i]
    return state_dict

def run_loop(model, gd, data, save_desc, lr=1e-3, start_epoch=0, epoch_block=10000, num_its=20, p_uncond=0.0, default_im='im_10.png', ddim_gd=None, latent_orthog=False, ema_rate=0.9999, dataset='clevr', downweight=False, image_size=64, use_dist=False):
    if use_dist: # distributed training
        from .util.dist_util import dev
        device = dev()
    else:
        device = 'cuda' if th.cuda.is_available() else 'cpu'

    # ddim sampling for generating samples per epoch block
    if ddim_gd == None:
        ddim_gd = create_ddim_diffusion(diffusion_defaults())
    
    # ema_params = copy.deepcopy(list(model.parameters()))
    ema_model = EMA(model, beta=ema_rate, update_every=1)
    ema_model.to('cuda')

    optimizer = th.optim.Adam(model.parameters(),
                                lr=lr)
                                # weight_decay = 1e-8)
    total_epochs = epoch_block * num_its

    save_dir = 'logs_' + save_desc # f'{save_desc}_params/'
    os.makedirs(save_dir, exist_ok=True)
    print(f'Saving model ckpts to {save_dir}')

    free = p_uncond > 0
    for epoch in range(start_epoch, start_epoch + total_epochs):
        batch, cond = next(data)
        batch = batch.to(device)
        model_kwargs = {}
    
        if not free:
            model_kwargs = dict(latent=None, x_start=batch)
        else:
            
            b = batch.shape[0]
            rand_values = th.rand(b, 1).to(device)
            keep_mask = rand_values >= p_uncond
            null_emb = th.zeros(b, model.latent_dim_expand).to(device)
            latent_emb = model.encode_latent(batch)
            emb = th.where(
                keep_mask,
                latent_emb,
                null_emb
            )
            model_kwargs = dict(latent=emb)

        t = uniform_sample_timesteps(gd.num_timesteps, len(batch)).to(device)

        loss = gd.training_losses(model, batch, t, model_kwargs=model_kwargs, latent_orthog=latent_orthog, downweight=downweight)
        loss = loss.mean() 
        
        # The gradients are set to zero,
        # the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update ema params
        # update_ema(ema_params, list(model.parameters()), ema_rate=ema_rate)
        ema_model.update()

        imgs_save_dir = f'gen_imgs_{save_desc}'
        if epoch % epoch_block == 0:
            print(f'img at {epoch} epochs')
            print(f'Saving images to {imgs_save_dir}')
            get_gen_images(model, ddim_gd, im_path=default_im, desc=str(epoch), save_dir=imgs_save_dir, free=free, dataset=dataset, sample_method='ddim', num_images=1, image_size=image_size)

            print('loss:')
            print(loss)

            th.save(model.state_dict(), os.path.join(save_dir, f'model_{epoch}.pt'))
            # save ema params
            # ema_state_dict = params_to_state_dict(ema_params, model)
            th.save(ema_model.state_dict(), os.path.join(save_dir, f'ema_{ema_rate}_{epoch}.pt'))
            

def create_ema(save_desc, epoch_block=10000, last_epoch=140000):
    save_dir = 'logs_' + save_desc
    defaults = model_defaults()

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--ema_rate', type=float, default=0.99)
    args = parser.parse_args()

    ema_rate = args.ema_rate
    model_kwargs = args_to_dict(args, model_defaults().keys())
    model = create_diffusion_model(**model_kwargs)
    model.eval()
    device = 'cuda'
    model.to(device)
    ema_params = 0

    for epoch in range(0, last_epoch + 1, epoch_block):
        ckpt_path = os.path.join(save_dir, f'model_{epoch}.pt')
        print(f'loading from {ckpt_path}')
        checkpoint = th.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        if epoch == 0:
            ema_params = copy.deepcopy(list(model.parameters()))
        update_ema(ema_params, list(model.parameters()), ema_rate=ema_rate)
        print(epoch)

        ema_state_dict = params_to_state_dict(ema_params, model)
        th.save(ema_state_dict, os.path.join(save_dir, f'ema_{ema_rate}_{epoch}.pt'))

if __name__=='__main__':
    save_desc = 'unet_model_celebahq_10000_xstart_emb_128'
    create_ema(save_desc)