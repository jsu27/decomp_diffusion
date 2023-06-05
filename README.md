Run following to create a conda environment, and activate it:
```
conda create -n decomp_diff python=3.8
conda activate decomp_diff
```

To install this package, clone this repository and then run:
```
pip install -e .
```

Training U-Net model on CelebaHQ:

```
DEVICE=$CUDA_VISIBLE_DEVICES
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
TRAIN_FLAGS="--model_desc unet_model --num_images 10000 --batch_size 16 --dataset celebahq --num_channels 64 --extra_desc dist"
CUDA_VISIBLE_DEVICES=$DEVICE python -m torch.distributed.run --nproc_per_node=$NUM_DEVICES image_train.py $TRAIN_FLAGS
```

Generating images:
```
python gen_image.py --ckpt_path $MODEL_CHECKPOINT --im_path $IM_PATH --save_dir $SAVE_DIR --model_desc unet_model --num_channels 64
```

