import torch as th
import pdb
import os
import pickle
import numpy as np
import random
import copy
from ema_pytorch import EMA

import util.logger as logger
import util.dist_util as dist_util

from gen_image import get_gen_images


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

def run_loop(model, gd, data, model_desc, save_desc, lr=1e-3, start_epoch=0, epoch_block=10000, num_its=20, p_uncond=0.0, default_im='im_10.png', latent_orthog=False, ema_rate=0.9999, dataset='clevr', downweight=False, image_size=64):
    # ema_params = copy.deepcopy(list(model.parameters()))
    ema_model = EMA(model, beta=ema_rate, update_every=1)
    ema_model.to('cuda')

    optimizer = th.optim.Adam(model.parameters(),
                                lr=lr)
                                # weight_decay = 1e-8)
    total_epochs = epoch_block * num_its
    outputs = []
    losses = []

    save_dir = 'logs_' + save_desc # f'{save_desc}_params/'
    os.makedirs(save_dir, exist_ok=True)
    print(f'Saving model ckpts to {save_dir}')

    free = p_uncond > 0
    num_uncond_epochs = 0
    for epoch in range(start_epoch, start_epoch + total_epochs):
        batch, cond = next(data)
        batch = batch.to(dist_util.dev())
        model_kwargs = {}
        if model_desc == 'encoded_autoencoder': # TODO deprecated
            model_kwargs['latent'] = None
        else:

            if not free:
                model_kwargs = dict(latent=None, x_start=batch)
            else:
                
                b = batch.shape[0]
                rand_values = th.rand(b, 1).to(dist_util.dev())
                keep_mask = rand_values >= p_uncond
                null_emb = th.zeros(b, model.latent_dim_expand).to(dist_util.dev())
                latent_emb = model.encode_latent(batch)
                emb = th.where(
                    keep_mask,
                    latent_emb,
                    null_emb
                )

                model_kwargs = dict(latent=emb)

        t = uniform_sample_timesteps(gd.num_timesteps, len(batch)).to(dist_util.dev())

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
            get_gen_images(model, gd, sample_method='ddpm', im_path=default_im, desc=str(epoch), save_dir=imgs_save_dir, free=free, dataset=dataset, num_images=2, image_size=image_size)

            print('loss:')
            print(loss)

            th.save(model.state_dict(), os.path.join(save_dir, f'model_{epoch}.pt'))
            # save ema params
            # ema_state_dict = params_to_state_dict(ema_params, model)
            th.save(ema_model.state_dict(), os.path.join(save_dir, f'ema_{ema_rate}_{epoch}.pt'))

            if epoch > 0:
                print('frac uncond epochs so far')
                print(num_uncond_epochs / epoch)
            

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