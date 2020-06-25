import hashlib
import os
import numpy as np
from numpy.lib.format import open_memmap
import tensorflow as tf



class DatasetBase:
    def __init__(self, memmap_directory=None, overwrite_memmap=False):
        self.memmap_directory = memmap_directory
        self.overwrite_memmap = overwrite_memmap
        self.memmap_file = None
        self.hash_data = None
        self.data = None
        self.shape = None
        self.dtype = None

    def read_or_create_data(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

        if isinstance(shape, list):
            raise ValueError("Shape must be a tuple, e.g. (100,32,32,3) not a list, e.g. [100,32,32,3]")

        if self.memmap_directory:
            if self.hash_data is None:
                raise ValueError("Please set the hash_data (as a numpy array) to use memory mapping")
            self.memmap_file = os.path.join(self.memmap_directory, self.get_hash_id() + ".npy")
            if self.overwrite_memmap is False and os.path.exists(self.memmap_file):
                print("Existing data found at {}".format(self.memmap_file))
                self.data = open_memmap(self.memmap_file, mode='r+', dtype=self.dtype, shape=self.shape)
                # Check if not all zeros
                # If all zeros, usually indication of an error creating the memmap file previously, therefore recreate
                if np.count_nonzero(self.data[0]) > 0:
                    return True
                else:
                    self.data._mmap.close()
                    print("File was likely corrupted, recreating.")
            print("Creating memmap file at {}".format(self.memmap_file))
            os.makedirs(self.memmap_directory, exist_ok=True)
            self.data = open_memmap(self.memmap_file, mode='w+', dtype=self.dtype, shape=self.shape)
        else:
            self.data = np.zeros(self.shape, dtype=self.dtype)

        return False

    def get_hash_id(self):
        """
        Creates a 16 character hash id based on the hash data. This is used to create a unique
        filename for the numpy memmap array
        :return: 16 character hash id
        """
        if self.hash_data is None:
            raise ValueError("Please set the hash_data before getting the hash")
        else:
            return hashlib.sha256(repr(self.hash_data).encode('UTF-8')).hexdigest()[0:16]


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#
#     # Test parsing
#     _filenames = parse_directory(r"D:\Datasets\Seagrass\SeagrassFrames")
#
#     # Test resizing
#     _im = load_image(r"../../test/example.jpg")
#     plt.imshow(_im)
#     plt.title(_im.shape)
#     plt.show()
#     _im = load_image(r"../../test/example.jpg", (600, 512), 'rgb')
#     plt.imshow(_im)
#     plt.show()
#     _im = load_image(r"../../test/example.jpg", (300, 512), 'rgb')
#     plt.imshow(_im)
#     plt.show()

    # Test loading
    # _gen = ImageLoader(_filenames)
    # _ims = _gen.load(max_images=10)
    # plt.imshow(_ims[0])
    # plt.title(_ims[0].shape)
    # plt.show()
    #
    # # Test loading into memmap
    # _ds = DatasetBase(memmap_directory=r"D:\temp", overwrite_memmap=True)
    # _ds.hash_data = np.asarray("testing loading into memmap")
    # _ds.read_or_create_data((10, 2600, 4624, 3), np.float32)
    # _gen = ImageLoader(_filenames)
    # _gen.load_into_array(_ds.data, max_images=10)
    # plt.imshow(_ds.data[0] / 255)
    # plt.title(_ds.data[0].shape)
    # plt.show()
    # plt.imshow(_ds.data[9] / 255)
    # plt.title(_ds.data[9].shape)
    # plt.show()

