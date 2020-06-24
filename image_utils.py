import multiprocessing
import numpy as np
from skimage import io as skio, transform as skt
from tensorflow.keras.utils import Sequence, OrderedEnqueuer
from tqdm import tqdm
#
#
#
#
# class ImageLoader(Sequence):
#     def __init__(self, filenames, img_size=None, img_type='rgb', prepro=None, batch_size=1):
#         if isinstance(filenames, dict):
#             self.filenames = [v for key, val in filenames.items() for v in val]
#         else:
#             self.filenames = filenames
#         self.batch_size = batch_size
#         self.img_size = img_size
#         self.img_type = img_type
#         self.prepro = prepro
#
#     def __data_generation(self, filenames):
#         images = []
#         for filename in filenames:
#             image = load_image(filename, self.img_size, self.img_type)
#             if self.prepro is not None:
#                 image = self.prepro(image)
#             images.append(image)
#         return filenames, np.asarray(images)
#
#     def __getitem__(self, index):
#         batch_filenames = self.filenames[index * self.batch_size:(index + 1) * self.batch_size]
#         filenames_and_images = self.__data_generation(batch_filenames)
#         return filenames_and_images
#
#     def __len__(self):
#         return int(np.ceil(len(self.filenames) / self.batch_size))
#
#     def load(self, workers=None, max_images=None):
#         images = []
#         if workers is None:
#             workers = np.min((multiprocessing.cpu_count(), 8))
#         enq = OrderedEnqueuer(self, use_multiprocessing=True)
#         enq.start(workers=workers, max_queue_size=multiprocessing.cpu_count() * 4)
#         gen = enq.get()
#         total = len(self)
#         if max_images is not None:
#             total = max_images
#         for i in tqdm(range(total)):
#             fns, ims = next(gen)
#             images.extend(ims)
#         enq.stop()
#         return images
#
#     def load_into_array(self, arr, workers=None, max_images=None):
#         total = len(self)
#         if max_images is not None:
#             total = max_images
#         if len(arr) != total:
#             raise ValueError("Length of array is not correct: length {}, required length {}".format(len(arr), len(self.filenames)))
#         if workers is None:
#             workers = np.min((multiprocessing.cpu_count(), 8))
#         enq = OrderedEnqueuer(self, use_multiprocessing=True)
#         enq.start(workers=workers, max_queue_size=multiprocessing.cpu_count() * 4)
#         gen = enq.get()
#         for i in tqdm(range(total)):
#             fns, ims = next(gen)
#             ims = ims.astype(arr.dtype)
#             arr[i * self.batch_size:(i+1)*self.batch_size] = ims
#         enq.stop()


def load_image(filename, img_size=None, img_type='rgb'):
    """
    Loads an image, converting it if necessary
    :param filename: Filename of image to load
    :param img_size: Size of image, e.g. (224, 224). If None, the image dimensions are preserved
    :param img_type: 'rgb' (colour) or 'k' (greyscale)
    :return:
    """
    # Colour
    if img_type == 'rgb':
        im = skio.imread(filename)
        # im = np.asarray(im, dtype=np.float)
        # If it was a single channel image, make into 3-channel
        if im.ndim == 2:
            im = np.expand_dims(im, -1)
            im = np.repeat(im, repeats=3, axis=-1)
    # Greyscale
    elif img_type == 'k' or img_type == 'greyscale':
        im = skio.imread(filename, as_gray=True)
        im = np.expand_dims(im, -1)
    else:
        raise ValueError("img_type must be 'rgb' or 'k'")
    # Resize and pad
    if img_size is not None:
        im = resize_and_pad_image(im, img_size)
    return im


def resize_and_pad_image(im, img_size):
    # Get the ratio of width to height for each
    current_whratio = im.shape[1] / im.shape[0]
    desired_whratio = img_size[1] / img_size[0]
    # Check if the image has roughly the same ratio, else pad it
    if np.round(im.shape[0] * desired_whratio) != im.shape[1]:
        height = im.shape[0]
        width = im.shape[1]
        # Desired shape is wider than current one
        if desired_whratio > current_whratio:
            half = np.round(height * desired_whratio)
            height_pad_start = 0
            height_pad_end = 0
            width_pad_start = int(abs(np.floor((width - half) / 2)))
            width_pad_end = int(abs(np.ceil((width - half) / 2)))
        # Desired shape is taller than current
        else:
            half = np.round(width / desired_whratio)
            height_pad_start = int(abs(np.floor((height - half) / 2)))
            height_pad_end = int(abs(np.ceil((height - half) / 2)))
            width_pad_start = 0
            width_pad_end = 0
        # Constant value to pad with
        consts = [np.median(np.concatenate((im[0, :, i], im[-1, :, i], im[:, 0, i], im[:, -1, i]))) for i in range(im.shape[2])]
        # Pad
        im = np.stack(
            [np.pad(im[:, :, c],
                    ((height_pad_start, height_pad_end), (width_pad_start, width_pad_end)),
                    mode='constant',
                    constant_values=consts[c])
             for c in range(im.shape[2])], axis=2)
    # Resize
    if im.shape[0] != img_size[0] or im.shape[1] != img_size[1]:
        im = skt.resize(im, img_size)
    return im


def tiles_from_image(im, tile_size, tile_step=None):
    if tile_step is None:
        tile_step = tile_size
    max_h = im.shape[0] - tile_size[0]
    max_w = im.shape[1] - tile_size[1]
    steps_h = np.arange(0, max_h, tile_step[0])
    steps_w = np.arange(0, max_w, tile_step[0])
    num_h = len(steps_h)
    num_w = len(steps_w)
    if np.ndim(im) == 2:
        array = np.zeros((num_h, num_w, *tile_size), dtype=im.dtype)
    else:
        array = np.zeros((num_h, num_w, *tile_size, im.shape[2]), dtype=im.dtype)
    for yi, y in enumerate(steps_h):
        for xi, x in enumerate(steps_w):
            array[yi, xi, ...] = im[y:y+tile_size[0], x:x+tile_size[1], ...]
    return array
