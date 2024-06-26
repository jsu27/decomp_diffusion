# Compositional Image Decomposition with Diffusion Models
## We propose Decomp Diffusion, an unsupervised approach that discovers compositional concepts from images, represented by diffusion models. 

![](sample_images/teaser_denoising.gif)


### [Project Page]() | [Paper]() | [Google Colab][composable-demo] | [Huggingface][huggingface-demo] | [![][colab]][composable-demo] [![][huggingface]][huggingface-demo]

<hr>

This is the official codebase for **Unsupervised Compositional Image Decomposition with Diffusion Models**.

[Compositional Image Decomposition with Diffusion Models]()
    <br>
    [Jocelin Su](https://github.com/jsu27) <sup>1*</sup>,
    [Nan Liu](https://nanliu.io) <sup>2*</sup>,
    [Yanbo Wang](https://openreview.net/profile?id=~Yanbo_Wang3) <sup>3*</sup>,
    [Joshua B. Tenenbaum](https://mitibmwatsonailab.mit.edu/people/joshua-tenenbaum/) <sup>1</sup>,
    [Yilun Du](https://yilundu.github.io) <sup>1</sup>,
    <br>
    <sup>*</sup> Equal Contribution
    <br>
    <sup>1</sup>MIT, <sup>2</sup>UIUC, <sup>3</sup> TU Delft
    <br>
   

[colab]: <https://colab.research.google.com/assets/colab-badge.svg>
[huggingface]: <https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue>
[composable-demo]: <https://colab.research.google.com/drive/103YlXU0Pcfx7ndU2ZTozBy15fVzkhHyl?usp=sharing>
[huggingface-demo]: <https://huggingface.co/spaces/jsu27/decomp-diffusion>

--------------------------------------------------------------------------------------------------------


The [demo](notebooks/demo.ipynb) [![][colab]][composable-demo] notebook shows how to train a model and perform experiments on decomposition, reconstruction, and recombination of factors on CLEVR, as well as recombination in multi-modal and cross-dataset settings. 

* The codebase is built upon [Improved-Diffusion](https://github.com/openai/improved-diffusion).
* This codebase provides both training and inference code.
--------------------------------------------------------------------------------------------------------

## Setup

Run the following to create and activate a conda environment:
```
conda create -n decomp_diff python=3.8
conda activate decomp_diff
```
To install this package, clone this repository and then run:

```
pip install -e .
```
--------------------------------------------------------------------------------------------------------

## Training

We use a U-Net model architecture. To train a model, we specify its parameters and training parameters as follows:
```
MODEL_FLAGS="--emb_dim 64 --enc_channels 128"
TRAIN_FLAGS="--batch_size 16 --dataset clevr --data_dir ../"
```

For distributed training, we run the following:
```
DEVICE=$CUDA_VISIBLE_DEVICES
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

python -m torch.distributed.run --nproc_per_node=$NUM_DEVICES scripts/image_train.py $MODEL_FLAGS $TRAIN_FLAGS
```
Otherwise, we run:
```
python scripts/image_train.py $MODEL_FLAGS $TRAIN_FLAGS --use_dist False
```

--------------------------------------------------------------------------------------------------------

## Inference 

To generate images, we use a trained model and run a sampling loop, where DDPM sampling or DDIM sampling is specified. We provide pre-trained models for various datasets below. For example, a pre-trained CLEVR model is provided [here](https://www.dropbox.com/s/bqpc3ymstz9m05z/clevr_model.pt).

To perform decomposition and reconstruction of an input image, run the following:
```
MODEL_CHECKPOINT="clevr_model.pt"
MODEL_FLAGS="--emb_dim 64 --enc_channels 128"
python scripts/gen_image_script.py --dataset clevr --ckpt_path $MODEL_CHECKPOINT $MODEL_FLAGS --im_path sample_images/clevr_im_10.png --save_dir gen_clevr_img/ --sample_method ddim
```

In addition, we can generate results for multiple images in a dataset:
```
python scripts/gen_image_script.py --gen_images 100 --dataset $DATASET --ckpt_path $MODEL_CHECKPOINT $MODEL_FLAGS --save_dir gen_many_clevr_imgs/
```

Decomp Diffusion can also compose discovered factors. To combine factors across 2 images, run:
```
python scripts/gen_image_script.py --combine_method slice --dataset $DATASET --ckpt_path $MODEL_CHECKPOINT $MODEL_FLAGS --im_path $IM_PATH --im_path2 $IM_PATH2 --save_dir $SAVE_DIR 
```

See `gen_image_script.py` for additional options such as generating additive combinations or cross-dataset combinations.

--------------------------------------------------------------------------------------------------------


## Datasets
See our paper for details on training datasets. Note that Tetris images are 32x32 instead of 64x64.

| Dataset | Link | 
| :---: | :---: | 
| CLEVR | [Link](https://www.dropbox.com/s/1uk59q8aembfirp/images_clevr.tar.gz)
| CLEVR Toy | [Link](https://www.dropbox.com/s/ajtvg1fmr2xec7b/clevr_toy.zip)
| Tetris | [Link](https://www.dropbox.com/s/l0wtsfzo6mzjxls/tetris_images_32.zip)
| CelebA-HQ 128x128 | [Link](https://www.dropbox.com/scl/fi/t14whi1jrs7aahrewoaew/celebahq_data128x128.zip?rlkey=xctq15n6wjzemb8piu9hl6bs8&dl=0)
| KITTI | [Link](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip)
| Virtual KITTI 2 | [Link](http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar)
| Falcor3D | [Link](https://drive.google.com/uc?export=download&id=1XAQfFK1x6cpN1eiovbP0hVfLTm5SsSoJ)
| Anime | [Link](https://gwern.net/crop#danbooru2019-portraits)

--------------------------------------------------------------------------------------------------------

## Models
See our paper for details on model parameters for each dataset. We provide links to pre-trained models below, as well as their non-default parameter flags. We used `--batch_size 32` during training.

| Model | Link | Model Flags
| :---: | :---: | :---: |
| CLEVR | [Link](https://www.dropbox.com/s/bqpc3ymstz9m05z/clevr_model.pt) | `--emb_dim 64 --enc_channels 128`
| CelebA-HQ | [Link](https://www.dropbox.com/s/687wuamoud4cs9x/celeb_model.pt) | `--enc_channels 128`
| Faces | [Link](https://www.dropbox.com/s/ia1ehqtpch4b2mz/faces_model.pt) | `--enc_channels 128`
| CLEVR Toy | [Link](https://www.dropbox.com/s/f90ogyqk7siedid/toy_model.pt) | `--emb_dim 64 --enc_channels 128`
| Tetris | | `--image_size 32 --num_components 3 --num_res_blocks 1 --enc_channels 64`
| VKITTI | | `--num_channels 64 --enc_channels 64 --emb_dim 256`
| Combined KITTI | | `--num_channels 64 --enc_channels 64 --emb_dim 256`
| Falcor3D | | `--num_channels 64 --emb_dim 32 --channel_mult 1,2`

<!-- 
## Citing our Paper

If you find our code useful for your research, please consider citing 

``` 

``` -->
