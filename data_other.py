import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from filenames_dataset import FilenamesDataset
from image_dataset import ImageDataset

W_LOW = 128 // 2
W_MED = 160 // 2
W_HIGH = 192 // 2

def mnist_x(x_orig, mdl_input_dims, is_training):
    # rescale to [0, 1]
    x_orig = tf.cast(x_orig, dtype=tf.float32) / 255

    # get common shapes
    height_width = mdl_input_dims[:-1]
    n_chans = mdl_input_dims[-1]

    # training transformations
    if is_training:
        np.array(x_orig.shape.as_list()[1:-1])
        x1 = tf.image.central_crop(x_orig, np.min(W_MED / np.array(x_orig.shape.as_list()[1:-1])))
        x2 = tf.image.random_crop(x_orig, tf.concat((tf.shape(x_orig)[:1], [W_MED, W_MED], [n_chans]), axis=0))
        # x2 = tf.image.random_crop(x_orig, tf.concat(([20, 20], [n_chans]), axis=0))
        x = tf.stack([x1, x2])
        x = tf.transpose(x, [1, 0, 2, 3, 4])
        i = tf.squeeze(tf.random.categorical([[1., 1.]], tf.shape(x)[0]))
        x = tf.map_fn(lambda y: y[0][y[1]], (x, i), dtype=tf.float32)
        x = tf.image.resize(x, height_width)

    # testing transformations
    else:
        x = tf.image.central_crop(x_orig, np.mean(W_MED / np.array(x_orig.shape.as_list()[1:-1])))
        x = tf.image.resize(x, height_width)

    return x


def mnist_gx(x_orig, mdl_input_dims, is_training, sample_repeats):
    # if not training, return a constant value--it will unused but needs to be same shape to avoid TensorFlow errors
    if not is_training:
        return tf.zeros((0,) + mdl_input_dims)

    # rescale to [0, 1]
    x_orig = tf.cast(x_orig, dtype=tf.float32) / 255

    # repeat samples accordingly
    x_orig = tf.tile(x_orig, [sample_repeats] + [1] * len(x_orig.shape.as_list()[1:]))

    # get common shapes
    height_width = mdl_input_dims[:-1]
    n_chans = mdl_input_dims[-1]

    # random rotation
    rad = 2 * np.pi * 25 / 360
    x_rot = tf.contrib.image.rotate(x_orig, tf.random.uniform(shape=tf.shape(x_orig)[:1], minval=-rad, maxval=rad))
    gx = tf.stack([x_orig, x_rot])
    gx = tf.transpose(gx, [1, 0, 2, 3, 4])
    i = tf.squeeze(tf.random.categorical([[1., 1.]], tf.shape(gx)[0]))
    gx = tf.map_fn(lambda y: y[0][y[1]], (gx, i), dtype=tf.float32)

    # random crops
    x1 = tf.image.random_crop(gx, tf.concat((tf.shape(x_orig)[:1], [W_LOW, W_LOW], [n_chans]), axis=0))
    x2 = tf.image.random_crop(gx, tf.concat((tf.shape(x_orig)[:1], [W_MED, W_MED], [n_chans]), axis=0))
    x3 = tf.image.random_crop(gx, tf.concat((tf.shape(x_orig)[:1], [W_HIGH, W_HIGH], [n_chans]), axis=0))
    gx = tf.stack([tf.image.resize(x1, height_width),
                   tf.image.resize(x2, height_width),
                   tf.image.resize(x3, height_width)])
    gx = tf.transpose(gx, [1, 0, 2, 3, 4])
    i = tf.squeeze(tf.random.categorical([[1., 1., 1.]], tf.shape(gx)[0]))
    gx = tf.map_fn(lambda y: y[0][y[1]], (gx, i), dtype=tf.float32)

    # apply random adjustments
    def rand_adjust(img):
        img = tf.image.random_brightness(img, 0.4)
        img = tf.image.random_contrast(img, 0.6, 1.4)
        if img.shape.as_list()[-1] == 3:
            img = tf.image.random_saturation(img, 0.6, 1.4)
            img = tf.image.random_hue(img, 0.125)
        return img

    gx = tf.map_fn(lambda y: rand_adjust(y), gx, dtype=tf.float32)

    return gx


# def configure_data_set(ds, info, batch_size, is_training, **kwargs):
#     """
#     :param ds: TensorFlow data set object
#     :param info: TensorFlow DatasetInfo object
#     :param batch_size: batch size
#     :param is_training: indicator to pre-processing function
#     :return: a configured TensorFlow data set object
#     """
#     # enable shuffling and repeats
#     ds = ds.shuffle(10 * batch_size, reshuffle_each_iteration=True).repeat(1)
#
#     # batch the data before pre-processing
#     ds = ds.batch(batch_size)
#
#     # pre-process the data set
#     with tf.device('/cpu:0'):
#         ds = pre_process_data(ds, info, is_training, **kwargs)
#
#     # enable prefetch
#     ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#
#     return ds
# def configure_data_set(ds, info, batch_size, is_training, **kwargs):
#     """
#     :param ds: TensorFlow data set object
#     :param info: TensorFlow DatasetInfo object
#     :param batch_size: batch size
#     :param is_training: indicator to pre-processing function
#     :return: a configured TensorFlow data set object
#     """
#     # enable shuffling and repeats
#     ds = ds.shuffle(10 * batch_size, reshuffle_each_iteration=True).repeat(1)
#
#     # batch the data before pre-processing
#     ds = ds.batch(batch_size)
#
#     # pre-process the data set
#     with tf.device('/cpu:0'):
#         ds = pre_process_data(ds, info, is_training, **kwargs)
#
#     # enable prefetch
#     ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#
#     return ds


class IICGenerator(object):
    def __init__(self,
                 data,
                 labels,
                 idxs=None,
                 batch_size=32,
                 shuffle=True,
                 prefetch=4,
                 is_training=True,
                 num_repeats=5,
                 one_shot=False,
                 data_dtype=tf.float32,
                 labels_dtype=tf.float32):
        self.data = data
        self.data_dtype = data_dtype
        self.data_shape = self.data.shape[1:]
        if isinstance(labels, list):
            labels = np.asarray(labels)
        self.labels = labels
        self.labels_dtype = labels_dtype
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prefetch = prefetch
        self.is_training = is_training
        self.num_repeats = num_repeats
        self.one_shot = one_shot
        if idxs is None:
            self.idxs = np.arange(len(data))
        else:
            self.idxs = idxs.copy()
        self.on_epoch_end()

    def __len__(self):
        if self.one_shot:
            return int(np.ceil(len(self.idxs) / self.batch_size))
        else:
            return int(np.floor(len(self.idxs) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def generator(self):
        i = 0
        while i < len(self.data):
            idx = self.idxs[i]
            if self.labels is None:
                yield self.data[idx]
            else:
                yield (self.data[idx], self.labels[idx])
            i += 1
        self.on_epoch_end()

    def create(self):
        ds = tf.data.Dataset.from_generator(self.generator,
                                            output_types=(self.data_dtype, self.labels_dtype),
                                            output_shapes=(self.data[0].shape, self.labels[0].shape))
        """
        From docs:
        Performance can often be improved by setting num_parallel_calls so that map will use multiple threads to process elements. 
        If deterministic order isn't required, it can also improve performance to set deterministic=False.

        Note that the map function has to take a Tensor input
        """
        print(self.data.shape)
        print(self.labels.shape)
        ds = ds.shuffle(10 * self.batch_size, reshuffle_each_iteration=True).repeat(1)
        # ds = ds.repeat(1)
        ds = ds.batch(self.batch_size)
        ds = ds.map(lambda x, y: {'x': mnist_x(x,
                                            mdl_input_dims=self.data_shape,
                                            is_training=self.is_training),
                               'gx': mnist_gx(x,
                                              mdl_input_dims=self.data_shape,
                                              is_training=self.is_training,
                                              sample_repeats=self.num_repeats),
                               'label': y},
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # # if self.one_shot is False:
        # #     ds = ds.repeat()
        # # ds = ds.repeat(1)
        # ds = ds.prefetch(self.prefetch)
        return ds

    @staticmethod
    def map_fn_divide_255(t):
        t = tf.cast(t, tf.float32)
        return tf.divide(t, 255.0)


def load(data_set_name, **kwargs):
    """
    :param data_set_name: data set name--call tfds.list_builders() for options
    :return:
        train_ds: TensorFlow Dataset object for the training data
        test_ds: TensorFlow Dataset object for the testing data
        info: data set info object
    """
    # get data and its info
    ds, info = tfds.load(name=data_set_name, with_info=True)

    # configure the data sets
    if 'train' in info.splits:
        train_ds = configure_data_set(ds=ds['train'], info=info, is_training=True, **kwargs)
    else:
        train_ds = None
    if 'test' in info.splits:
        test_ds = configure_data_set(ds=ds['test'], info=info, is_training=False, **kwargs)
    else:
        test_ds = None
    return train_ds, test_ds, info


def load_from_directory(source_dir, split, batch_size, num_repeats, subsample=1, memmap_directory=None):
    fs = FilenamesDataset(source_dir)
    fs.split(split)
    ds_train = ImageDataset(fs.train_filenames, fs.train_cls, subsample=subsample, memmap_directory=memmap_directory)
    ds_train.load()
    ds_test = ImageDataset(fs.test_filenames, fs.test_cls, subsample=subsample, memmap_directory=memmap_directory)
    ds_test.load()
    tf_train = IICGenerator(ds_train.data,
                            ds_train.cls,
                            batch_size=batch_size,
                            num_repeats=num_repeats,
                            prefetch=16,
                            is_training=True)
    tf_test = IICGenerator(ds_train.data,
                           ds_train.cls,
                           batch_size=batch_size,
                           num_repeats=num_repeats,
                           prefetch=16,
                           is_training=False)
    return tf_train.create(), tf_test.create(), fs.num_classes


if __name__ == "__main__":
    train, test, num_classes = load_from_directory("/media/mar76c/DATA/Data/Seagrass/SeagrassFramesPatches",
                                                   0.2,
                                                   32,
                                                   5,
                                                   "~/tmp")
    import matplotlib.pyplot as plt
    sess = tf.InteractiveSession()
    iterator = tf.compat.v1.data.make_initializable_iterator(train)
    sess.run(iterator.initializer)
    r = sess.run(iterator.get_next())
    plt.matshow(r['x'][-1])
    plt.matshow(r['gx'][-1])
