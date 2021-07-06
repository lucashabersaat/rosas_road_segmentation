import torch
import cv2

from common.util import np_to_tensor
from common.read_data import *
from common.plot_data import *


class ImageDataSet(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index

    def __init__(self, path, device, use_patches=True, resize_to=(400, 400), divide_into_four=True):
        self.path = path

        self.device = device
        self.use_patches = use_patches
        self.resize_to = resize_to
        self.x, self.y, self.n_samples = None, None, None
        self._load_data()

        if divide_into_four:
            self._divide_into_four()

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = load_all_from_path(os.path.join(self.path, "images"))
        self.y = load_all_from_path(os.path.join(self.path, "groundtruth"))

        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x, self.y)
        else:
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x])
            self.y = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.y])
        self.x = np.moveaxis(self.x, -1, 1)
        self.n_samples = len(self.x)

    def _divide_into_four(self):
        n, c, w, h = self.x.shape
        assert (n, w, h) == self.y.shape

        # half width and height
        w_2 = w // 2
        h_2 = h // 2

        # divide into the four quadrant and concatenate
        x1 = self.x[:, :, :w_2, :h_2]
        x2 = self.x[:, :, w_2:, :h_2]
        x3 = self.x[:, :, :w_2, h_2:]
        x4 = self.x[:, :, w_2:, h_2:]
        self.x = np.concatenate([x1, x2, x3, x4])

        y1 = self.y[:, :w_2, :h_2]
        y2 = self.y[:, w_2:, :h_2]
        y3 = self.y[:, :w_2, h_2:]
        y4 = self.y[:, w_2:, h_2:]
        self.y = np.concatenate([y1, y2, y3, y4])

    def _preprocess(self, x, y):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing

        s = torch.std(x, [1, 2])

        for i in range(0, x.shape[1], 16):
            for j in range(0, x.shape[1], 16):
                m = torch.mean(x[:, i: i + 16, j: j + 16], [1, 2])

                x[0, i: i + 16, j: j + 16] /= m[0]
                x[1, i: i + 16, j: j + 16] /= m[1]
                x[2, i: i + 16, j: j + 16] /= m[2]

        x[0] /= s[0]
        x[1] /= s[1]
        x[2] /= s[2]

        return x, y

    def __getitem__(self, item):
        return self._preprocess(
            np_to_tensor(self.x[item], self.device),
            np_to_tensor(self.y[[item]], self.device),
        )

    def __len__(self):
        return self.n_samples


class TestImageDataSet(ImageDataSet):
    # dataset class that deals with loading the data and making it available by index

    def __init__(self, path, device, use_patches=True, resize_to=(400, 400), divide_into_four=True):
        super(TestImageDataSet, self).__init__(path, device, use_patches, resize_to, divide_into_four)

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = load_all_from_path(self.path)
        self.size = self.x.shape[1:3]

        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x)
        else:
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x])
        self.x = np.moveaxis(self.x, -1, 1)
        self.n_samples = len(self.x)

    def _divide_into_four(self):
        n, c, w, h = self.x.shape

        # half width and height
        w_2 = w // 2
        h_2 = h // 2

        # divide into the four quadrant and concatenate
        x1 = self.x[:, :, :w_2, :h_2]
        x2 = self.x[:, :, w_2:, :h_2]
        x3 = self.x[:, :, :w_2, h_2:]
        x4 = self.x[:, :, w_2:, h_2:]
        self.x = np.concatenate([x1, x2, x3, x4])
        self.n_samples = len(self.x)

    @staticmethod
    def put_back(x):
        """The inverse operation of the dividing into four."""
        x = np.concatenate(x, 0)
        n, c, w, h = x.shape
        n_4 = n // 4

        result = np.zeros([n_4, 1, w * 2, h * 2])
        result[:, :, :w, :h] = x[:n_4, :, :, :]
        result[:, :, w:, :h] = x[n_4:n_4*2, :, :, :]
        result[:, :, :w, h:] = x[2*n_4:3*n_4, :, :, :]
        result[:, :, w:, h:] = x[3*n_4:4*n_4, :, :, :]

        result = np.moveaxis(result, -1, 1)  # CHW to HWC
        result = result.squeeze()
        return result

    def _preprocess(self, x, y):
        raise NameError('Preprocessing for Test Data?')

    def __getitem__(self, item):
        return np_to_tensor(self.x[item], self.device)

    def __len__(self):
        return self.n_samples
