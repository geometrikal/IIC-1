import numpy as np
from sklearn.model_selection import train_test_split
from mml.data.utils import parse_directory, parse_csv


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
