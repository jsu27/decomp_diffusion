import torch as th
from torch.utils.data import DataLoader, Dataset

from glob import glob
from imageio import imread
from skimage.transform import resize as imresize

# import tetrominoes
# import tensorflow.compat.v1 as tf


def get_dataset(dataset_type, base_dir, start_index=0, num_images=None, resolution=64):
    """Get dataset class"""
    DATASET_MAPPING = dict(
        clevr=Clevr,
        clevr_toy=ClevrToy,
        celebahq=CelebaHQ,
        falcor3d=Falcor3d,
        kitti=Kitti,
        vkitti=VirtualKitti,
        comb_kitti=CombinedKitti,
        tetris=Tetrominoes,
        anime=Anime,
        faces=Faces
    )
    if dataset_type in DATASET_MAPPING:
        dataset = DATASET_MAPPING[dataset_type](base_dir, start_index=start_index, num_images=num_images, resolution=resolution)
    else:
        raise NotImplementedError(f'dataset: {dataset_type} is not implemented.')
    return dataset


class Data(Dataset):
    def __init__(self, base_dir, path='', resolution=64, start_index=0, num_images=None):
        self.resolution = resolution
        self.base_dir = base_dir
        self.path = self.base_dir + path
        self.images = sorted(glob(self.path))
        self.start_index = start_index
        self.num_images = num_images
        if num_images is not None:
            self.images = self.images[start_index:start_index + num_images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im_path = self.images[index]
        im = imread(im_path)
        im = imresize(im, (self.resolution, self.resolution))[:, :, :3]

        im = th.Tensor(im).permute(2, 0, 1)
        return im, index


class Clevr(Data):
    def __init__(self, base_dir, resolution=64, start_index=0, num_images=None):
        super().__init__(base_dir, path='images_clevr/*.png', resolution=resolution, start_index=start_index, num_images=num_images)


class ClevrToy(Data):
    def __init__(self, base_dir, resolution=64, start_index=0, num_images=None):
        super().__init__(base_dir, path='clevr_toy/*.png', resolution=resolution, start_index=start_index, num_images=num_images)


class CelebaHQ(Data):
    def __init__(self, base_dir, resolution=64, start_index=0, num_images=None):
        super().__init__(base_dir, path='celebahq/data128x128/*.jpg', resolution=resolution, start_index=start_index, num_images=num_images)


class Falcor3d(Data):
    def __init__(self, base_dir, resolution=64, start_index=0, num_images=None):
        super().__init__(base_dir, path='Falcor3D_down128/images/*.png', resolution=resolution, start_index=start_index, num_images=num_images)


class Kitti(Data):
    def __init__(self, base_dir, resolution=64, num_images=None, start_index=0):
        super().__init__(base_dir, path='kitti_data_tracking_image_2/training/image_02/*/*.png', resolution=resolution, start_index=start_index, num_images=num_images)

    def __getitem__(self, index):
        # 433 - 808
        im = self.images[index]
        im = imread(im)
        im = im[:, 433:808, :]
        im = imresize(im, (self.resolution, self.resolution))[:, :, :3]

        im = th.Tensor(im).permute(2, 0, 1)
        return im, index


class VirtualKitti(Data):
    def __init__(self, base_dir, resolution=64, num_images=None, start_index=0):
        super().__init__(base_dir, path='vkitti_2.0.3_rgb/*/*/frames/rgb/Camera_0/*.jpg', resolution=resolution, start_index=start_index, num_images=num_images)

    def __getitem__(self, index):
        # 433 - 808
        im = self.images[index]
        im = imread(im)
        im = im[:, 433:808, :]
        im = imresize(im, (self.resolution, self.resolution))[:, :, :3]

        im = th.Tensor(im).permute(2, 0, 1)
        return im, index


class CombinedKitti(Data):
    def __init__(self, base_dir, resolution=64, start_index=0, num_images=None):
        self.base_dir = base_dir 
        self.path1 = self.base_dir + 'kitti_data_tracking_image_2/training/image_02/*/*.png'
        self.path2 = self.base_dir + 'vkitti_2.0.3_rgb/*/*/frames/rgb/Camera_0/*.jpg'

        images1 = sorted(glob(self.path1)) # less kitti data, 8008
        images2 = sorted(glob(self.path2)) # 21260

        images = images1 * 3 + images2
        self.images = images
        if num_images is not None:
            self.images = self.images[:num_images]
        self.resolution = resolution

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        # 433 - 808
        im = self.images[index]
        im = imread(im)
        im = im[:, 433:808, :]
        im = imresize(im, (self.resolution, self.resolution))[:, :, :3]

        im = th.Tensor(im).permute(2, 0, 1)
        return im, index

# class TetrominoesLoader():

#     def __init__(self, resolution=32, batch_size=16, num_images=None):
#         self.resolution = resolution # real resolution 35
#         self.base_dir = '/om2/user/jocelin/'
#         tf_records_path = self.base_dir + 'tetrominoes_train.tfrecords'

#         dataset = tetrominoes.dataset(tf_records_path)
#         batched_dataset = dataset.batch(batch_size)  # optional batching
#         iterator = batched_dataset.make_one_shot_iterator()
#         self.data = iterator.get_next()
#         config = tf.ConfigProto(
#                 device_count = {'GPU': 0}
#             )
#         self.sess = tf.InteractiveSession(config=config)

#     def __iter__(self):
#         return self

#     def __next__(self):
#         d = self.sess.run(self.data)
#         img = d['image']
#         img = img.transpose((0, 3, 1, 2))
#         img = img / 255.
#         img = imresize(img, (img.shape[0], img.shape[1], self.resolution, self.resolution))
#         img = th.Tensor(img).contiguous()

#         return img, th.ones(1)

#     def __len__(self):
#         return 1e6 if num_images == None else num_images

class Tetrominoes(Data):
    def __init__(self, base_dir, resolution=32, start_index=0, num_images=None):
        super().__init__(base_dir, path='tetris_images_32/*.png', resolution=resolution, start_index=start_index, num_images=num_images)


class Anime(Data):
    def __init__(self, base_dir, resolution=64, start_index=0, num_images=None):
        super().__init__(base_dir, path='anime_portraits/*.jpg', resolution=resolution, start_index=start_index, num_images=num_images)


class Faces(Data):
    def __init__(self, base_dir, resolution=64, num_images=None, start_index=0):
        self.base_dir = base_dir
        self.path1 = self.base_dir + 'celebahq/data128x128/*.jpg'
        self.path2 = self.base_dir + 'anime_portraits/*.jpg'
        self.resolution = resolution
        self.images1 = sorted(glob(self.path1))
        self.images2 = sorted(glob(self.path2))
        if num_images is not None:
            self.images = self.images1[:num_images // 2] + self.images2[:num_images // 2]
        else: 
            self.images = self.images1 + self.images2



def load_data(
    *,
    base_dir,
    dataset_type,
    batch_size,
    image_size,
    deterministic=False,
    random_crop=False,
    random_flip=False,
    num_images=None
):  

    dataset = get_dataset(dataset_type, base_dir=base_dir, num_images=num_images, resolution=image_size)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True
        )
    while True:
        yield from loader
