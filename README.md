# Road Segmentation

_2021 | ETH Zürich | D-INFK | Computational Intelligence Lab | Course Project_

by **RO**bin Bisping, **SA**muel Bedassa and Luc**AS** Habersaat

## Idea

We combine the UNet model, together with Transformer, as done before in this [paper](https://arxiv.org/abs/2102.04306)
to segment aerial images into road and non-road. This has been done only for medical image segmentation. The strength of
recognizing global dependencies of Transformer complement this very weakness of UNets and vice versa. We argue, that
this can also be applied on road segmentation.

## Get Started

### Installation

To install all required packages:

```
$ pip install -r requirements.txt
```
### Run

All main features are run in  `main.py`. You can either train a model or load one, that has been trained before. Both
times, it will predict using the test data in `data/test_images` and save the submission in `data/submissions`


#### Train

To train a model, use the `-train` argument with a valid model name. For instance:

```
$ python main.py -train unet_transformer
```

To see what models are available, checkout `common/get_model.py`. 

#### Load

To load a model, that has been trained before, use the `-load` argument with a number that indicates the version. For
instance:

```
$ python main.py -load 25
```

If no number is given, it will take the latest.
All the trained models and the respective version number can be found in `data/lightning_logs`. Train a model and it
will be logged there.

#### Pretrained Data
The TransUNet model uses pretrained data from [here](https://github.com/google-research/vision_transformer) that can be downloaded [here](https://console.cloud.google.com/storage/vit_models/
). Save this in `data/models/imagenet`.

## Structure

Code used in multiple places is placed in `common`.
All neural network models are in `models`

All the data is in the `data` folder:

* **training** and the **validation** data set,
* the **test images** to make the predictions at the end,
* the **submission** files with the predictions
* lightning log files (not in version control)
* Ray Tune result log files (use Tensorboard on this directory)

## Logging

The PyTorch Lightning framework logs automatically for tensorboard. Each run is saved
in `data/lightning_logs`. You can start Tensorboard and analyzing the versions by running following in the console at
the root directory:

```
$ tensorboard --logdir data/lightning_logs
```

## References
* https://github.com/HXLH50K/U-Net-Transformer
* https://github.com/Beckschen/TransUNet
* https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets