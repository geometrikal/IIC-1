import numpy as np
import skimage.io as skio
import skimage.transform as skt
from mml.data.dataset import DatasetBase
from mml.data.image_loader import ParallelImageLoader


def rescale(im, factor):
    if np.ndim(im) <= 3:
        im = skt.rescale(im, [factor, factor, 1])
    else:
        im = skt.rescale(im, [1, factor, factor, 1])
    return (im * 255).astype(np.uint8)


class ImageDataset(DatasetBase):
    def __init__(self,
                 filenames,
                 cls,
                 rescale=None,
                 img_size=None,
                 memmap_directory=None,
                 overwrite_memmap=False):
        self.filenames = filenames
        self.cls = cls
        self.rescale = rescale
        self.dtype = np.uint8

        super().__init__(memmap_directory=memmap_directory, overwrite_memmap=overwrite_memmap)
        self.hash_data = ["ImageDataset", self.filenames, self.rescale, str(self.dtype)]
        print("Dataset id: {}".format(self.get_hash_id()))

        if img_size is None:
            # Read first image to see the size
            im = skio.imread(self.filenames[0])
            if self.rescale is not None:
                im = skt.rescale(im, [rescale, rescale, 1])
            self.img_size = im.shape
        else:
            self.img_size = img_size
        self.arr_size = (len(self.filenames), ) + self.img_size
        print("Array size is {}".format(self.arr_size))

    def load(self):
        if self.read_or_create_data(self.arr_size, np.uint8) is not True:
            if self.rescale is not None:
                loader = ParallelImageLoader(self.filenames, self.data, transform_fn=rescale, transform_args=(self.rescale, ))
            else:
                loader = ParallelImageLoader(self.filenames, self.data)
            loader.load()


if __name__ == '__main__':
    from mml.data.filenames_dataset import FilenamesDataset

    fs = FilenamesDataset(r"D:\Datasets\Seagrass\SeagrassFrames")
    fs.split(0.1)
    ds = ImageDataset(fs.test_filenames,
                      fs.test_cls,
                      memmap_directory=r"C:\temp",
                      overwrite_memmap=True)
    ds.load()
    import matplotlib.pyplot as plt
    for i in range(15):
        plt.imshow(ds.data[i])
        plt.show()
