import os

import torchvision.datasets as datasets


class TrainSplitImageFolder(datasets.ImageFolder):
    def __init__(self, root, train_txt=None, *args, **kwargs):
        if train_txt is None or not os.path.isfile(train_txt):
            print(f"Invalid filename {train_txt}, falling back to torchvision.datasets.ImageFolder")
            super().__init__(root, *args, **kwargs)
        else:
            with open(train_txt, 'r') as f:
                self.train_imgs = dict.fromkeys(
                    [img for img in f.read().split('\n') if img != ''], True)
            super().__init__(root, is_valid_file=self.is_train_img, *args, **kwargs)

    def is_train_img(self, filename):
        return self.train_imgs.get(filename, False)
