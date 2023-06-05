from .diffusion.gaussian_diffusion import get_named_beta_schedule, GaussianDiffusion
from .model.unet import UNetModel


def create_unet_model(
        image_size=64,
        num_channels=64, # 128, #192,
        enc_channels=64,
        num_res_blocks=2, # 3,
        num_components=4, 
        channel_mult="",
        num_heads=1,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.1,
        # text_ctx=128,
        # xf_width=512,
        # xf_layers=16,
        # xf_heads=8,
        # xf_final_ln=True,
        # xf_padding=True,
        # diffusion_steps=1000,
        # noise_schedule="squaredcos_cap_v2",
        # timestep_respacing="",
        use_scale_shift_norm=False, # True, # False??
        resblock_updown=True,
        model_desc='unet_model',
        emb_dim=256
        # dataset='clevr'
    ):
        # everything else False

    if channel_mult == "":
        if image_size == 64:
            channel_mult = (1, 2, 3) # (1, 2, 3, 4)
        elif image_size < 64: # eg 35
            channel_mult = (1, 2)
    elif len(channel_mult) > 0: # passed in comma-delimited series of numbers
        channel_mult = channel_mult.split(',')
        channel_mult = [int(n) for n in channel_mult]
        channel_mult = tuple(channel_mult)
        print('channel_mult')
        print(channel_mult)

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    unet = UNetModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=3, # 6,
        num_res_blocks=num_res_blocks,
        num_components=num_components,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        # num_classes=num_classes,
        # use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        encoder_channels=None,
        image_size=image_size,
        emb_dim=emb_dim
        # dataset=dataset
    )
    return unet

def unet_model_defaults():
    return dict(
        image_size=64,
        num_channels=64, # 128, #192,
        enc_channels=64,
        num_res_blocks=2, # 3,
        num_components=4, 
        channel_mult="",
        num_heads=1,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.1,
        # text_ctx=128,
        # xf_width=512,
        # xf_layers=16,
        # xf_heads=8,
        # xf_final_ln=True,
        # xf_padding=True,
        # diffusion_steps=1000,
        # noise_schedule="squaredcos_cap_v2",
        # timestep_respacing="",
        use_scale_shift_norm=False, # True, # False??
        resblock_updown=True,
        model_desc='unet_model',
        emb_dim=256
        # dataset='clevr'
    )

def create_diffusion_model(model_desc='unet_model', **model_kwargs):
    if model_desc == 'unet_model':
        model = create_unet_model(**model_kwargs)
    return model


def create_gaussian_diffusion(steps=1000, noise_schedule="squaredcos_cap_v2", predict_xstart=True):
    betas = get_named_beta_schedule(noise_schedule, steps)
    gd = GaussianDiffusion(betas, predict_xstart=predict_xstart)
    return gd

def model_defaults():
    return dict(
        in_channels=3,
        filter_dim=16,
        emb_dim=128,
        num_components=4,
        model_desc='decomp_diffusion',
        image_size=64 # added
    )

def diffusion_defaults():
    return dict(
        steps=1000,
        noise_schedule="squaredcos_cap_v2",
        predict_xstart=True
    )


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}
