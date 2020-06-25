import numpy as np
from sklearn.model_selection import train_test_split
import glob
import os

from collections import OrderedDict

def list_directory(source_dir):
    return sorted(glob.glob(os.path.join(source_dir, "*")))


def parse_directory(source_dir, verbose=True, skip='~'):
    """
    Parses a directory consisting of subdirectories named by class and stores the image filenames in a dictionary
    :param source_dir: The directory to parse
    :param verbose: Print out the progress
    :param skip: If a subdirectory starts with this character it will be skipped ('_' and '.' are skipped always)
    :return: Dictionary with class names for the keys and lists of filenames for the values
    """
    sub_dirs = sorted(glob.glob(os.path.join(source_dir, "*")))
    filenames = OrderedDict()
    if verbose:
        print("Parsing directory {}".format(source_dir))
    for sub_dir in sub_dirs:
        if os.path.isdir(sub_dir) is False:
            continue
        # Get the directory name
        sub_name = os.path.basename(sub_dir)
        # Skip directories starting with ~
        if sub_name.startswith(skip) or \
            sub_name.startswith('_') or \
            sub_name.startswith('.'):
            continue
        # Get the files
        if verbose:
            print("- {}".format(sub_name), end='')
        files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]:
            sub_files1 = sorted(glob.glob(os.path.join(sub_dir, ext)))
            sub_files2 = sorted(glob.glob(os.path.join(sub_dir, ext.upper())))
            files.extend(sub_files1)
            files.extend(sub_files2)
        # Add to dictionary
        if verbose:
            print(" ({} files)".format(len(files)))
        filenames[sub_name] = files
    return filenames


def flatten_to_list(filenames_dict: dict):
    filenames = []
    for value in filenames_dict.values():
        filenames.extend(value)
    return filenames


def parse_csv(csv_file, source_dir, file_idx=0, cls_idx=1, cls_label_idx=2, verbose=True):
    if verbose:
        print("Parsing csv file {} for directory {}".format(csv_file, source_dir))
    df = pd.read_csv(csv_file)
    num_classes = np.max(df.iloc[:, cls_idx]) + 1
    cls_labels = [df.loc[df.iloc[:, cls_idx] == i].iloc[0, cls_label_idx] for i in range(num_classes)]
    filenames = OrderedDict()
    for i in range(num_classes):
        if verbose:
            print("- {}".format(cls_labels[i]), end='')
        names = df.loc[df.iloc[:, cls_idx] == i].iloc[:, file_idx]
        paths = [os.path.join(source_dir, n) for n in names]
        filenames[cls_labels[i]] = paths
        if verbose:
            print(" ({} files)".format(len(paths)))
    return filenames

class FilenamesDataset(object):
    def __init__(self,
                 source_dir,
                 csv_file=None,
                 csv_idxs=[0,1,2]):
        self.source_dir = source_dir
        self.csv_file = csv_file
        self.train_filenames = None
        self.train_cls = None
        self.test_filenames = None
        self.test_cls = None

        if self.csv_file is None:
            self.cls_filenames = parse_directory(source_dir)
        else:
            self.cls_filenames = parse_csv(self.csv_file, self.source_dir, csv_idxs[0], csv_idxs[1], csv_idxs[2])
        self.cls_labels = list(self.cls_filenames.keys())
        self.cls = np.asarray([self.cls_labels.index(key) for key, val in self.cls_filenames.items() for v in val])
        self.filenames = [v for key, val in self.cls_filenames.items() for v in val]
        self.num_classes = len(self.cls_filenames.keys())

        if len(self.filenames) == 0:
            raise ValueError("Source directory does not contain any sub-directories with images")

    def split(self, test_size, stratify=True, seed=0):
        if stratify:
            stratify = self.cls
        else:
            stratify = None
        self.train_filenames, self.test_filenames, self.train_cls, self.test_cls = \
            train_test_split(self.filenames, self.cls, test_size=test_size, stratify=stratify, shuffle=True, random_state=seed)


if __name__ == '__main__':
    ds = FilenamesDataset(r"D:\Datasets\Seagrass\SeagrassFrames")
    ds.split(0.2)
    print(ds.train_filenames[0:10])
    print(ds.train_cls[0:10])
    print(ds.test_filenames[0:10])
    print(ds.test_cls[0:10])
