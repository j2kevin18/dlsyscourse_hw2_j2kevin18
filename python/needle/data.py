import numpy as np
import struct, gzip
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img == True:
            return np.flip(img, axis=1)
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        if shift_x <= 0:
            clip_x_up = shift_x + self.padding
            clip_x_down = clip_x_up + img.shape[0]
        else:
            clip_x_down = shift_x + self.padding + img.shape[0]
            clip_x_up = clip_x_down - img.shape[0]
        if shift_y <= 0:
            clip_y_up = shift_y + self.padding
            clip_y_down = clip_y_up + img.shape[1]
        else:
            clip_y_down = shift_y + self.padding + img.shape[1]
            clip_y_up = clip_y_down - img.shape[1]

        pad_img = np.pad(img, self.padding, 'constant')
        # print((clip_x_up, clip_x_down, clip_y_up, clip_y_down))
        return pad_img[clip_x_up : clip_x_down, clip_y_up : clip_y_down, self.padding:self.padding+img.shape[2]]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            shuffle_dataset = np.random.permutation(len(self.dataset))
            # print(shuffle_dataset)
            self.ordering = np.array_split(shuffle_dataset, 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        self.num_batch = len(self.ordering)
        self.iterNum = 0
        # print(self.ordering[self.num_batch-1].tolist())
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.iterNum < self.num_batch:
            # print(self.ordering[self.iterNum].tolist())
            batch_prec = self.dataset[self.ordering[self.iterNum].tolist()]
            if len(batch_prec) == 2:
                batch_X, batch_y = batch_prec
                batch = (Tensor(batch_X), Tensor(batch_y))
            elif len(batch_prec) == 1:
                batch_X, = batch_prec
                batch = (Tensor(batch_X), )
            self.iterNum = self.iterNum + 1
            return batch
        else:
            raise StopIteration 
        ### END YOUR SOLUTION

    # 我自己加的
    def __len__(self) -> int:
        return self.num_batch

def un_gz(filename):
  #解压缩.gz文件
  f_name = filename.replace(".gz", "")
  g_file = gzip.GzipFile(filename)
  open(f_name, "wb+").write(g_file.read())
  g_file.close()


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        path = '/content/drive/MyDrive/10714/hw2/'
        un_gz(path+label_filename)
        un_gz(path+image_filename)
        ungz_label_filename = label_filename.replace(".gz", "")
        ungz_image_filename = image_filename.replace(".gz", "")
        with open(path+ungz_label_filename, mode='rb') as labels, open(path+ungz_image_filename, mode='rb') as images:
            image_content = images.read(struct.calcsize('!4i'))
            _, image_num, image_height, image_width = struct.unpack("!4i", image_content)
            loaded = np.fromfile(file=images, dtype=np.uint8)

            max = loaded.max()
            min = loaded.min()
            X = loaded.reshape((image_num, image_height, image_width, 1)).astype(np.float32)
            #min-max normalization
            self.X = (X - min) / (max - min)
            
            label_content = labels.read(struct.calcsize('!2i'))
            _, label_num = struct.unpack("!2i", label_content)
            loaded = np.fromfile(file=labels, dtype=np.uint8)
            self.y = loaded.reshape((label_num,))

        self.transforms = transforms
        
        labels.close()
        images.close()
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        if isinstance(index, slice):
            # print(index)
            img_slice = self.X[index.start:index.stop:index.step]
            y_slice = self.y[index.start:index.stop:index.step]
            if self.transforms is not None:
                for transform in self.transforms:
                    img_slice = [transform(img) for img in img_slice]
            return (img_slice, y_slice)
        elif isinstance(index, list):
            img_slice = self.X[index]
            y_slice = self.y[index]
            if self.transforms is not None:
                for transform in self.transforms:
                    img_slice = [transform(img) for img in img_slice]
            return (img_slice, y_slice)
        else:
            img = self.X[index]
            if self.transforms is not None:
                for transform in self.transforms:
                    img = transform(img)
            return (img, self.y[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
