import torch

from common.util import np_to_tensor
from common.read_data import *


class ImageDataSet(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index

    def __init__(self, path, device, use_patches=True, resize_to=(400, 400)):
        self.path = path

        self.device = device
        self.use_patches = use_patches
        self.resize_to = resize_to
        self.x, self.y, self.n_samples = None, None, None
        self._load_data()

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = load_all_from_path(os.path.join(self.path, "images"))
        self.y = load_all_from_path(os.path.join(self.path, "groundtruth"))

        #a,b,c,d = self.x.shape
        #print(self.x.shape)
        #self.x = self.x.reshape(4*a,int(b/2),int(b/2),3)
        #print(self.x.shape)

        e,f,g = self.y.shape
        self.y = self.y.reshape(4*e,int(f/2),int(g/2))


        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x, self.y)
        else:
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x])
            self.y = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.y])
        self.x = np.moveaxis(self.x, -1, 1)
        self.n_samples = len(self.x)

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

    def __init__(self, path, device, use_patches=True, resize_to=(400, 400)):
        super(TestImageDataSet, self).__init__(path, device, use_patches, resize_to)

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = load_all_from_path(self.path)
        a,b,c,d = self.x.shape
        self.x = self.x.reshape(4*a,int(b/2),int(b/2),3)

        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x)
        else:
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x])
        self.x = np.moveaxis(self.x, -1, 1)
        self.n_samples = len(self.x)

    def _preprocess(self, x, y):
        raise NameError('Preprocessing for Test Data?')

    def __getitem__(self, item):
        return np_to_tensor(self.x[item], self.device)

    def __len__(self):
        return self.n_samples

if __name__ == "__main__":
    print('image data set preproc')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resize_to = 192*4
    batch_size = 4

    train_dataset = ImageDataSet(
        "data/training", device, use_patches=False, resize_to=(resize_to, resize_to)
    )
    val_dataset = ImageDataSet(
        "data/validation", device, use_patches=False, resize_to=(resize_to, resize_to)
    )




    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
