# Road Segmentation

_2021 | ETH Zürich | D-INFK | Computational Intelligence Lab | Course Project_

by **RO**bin Bisping, **SA**muel Bedassa and Luc**AS** Habersaat

## Installation

To install all required packages:

```
$ pip install -r requirements.txt
```

## Get Started

The different methods are in the `methods` folder and just running them will get you started. However, as lightning is used, it is recommended to run `methods/lightning/main.py` with parameters. See below.

Code used in multiple places is placed in `common`.

All the data is in the `data` folder:

* **training** and the **validation** data set,
* the **test images** to make the predictions at the end,
* the **submission** files with the predictions
* lightning log files (not in version control)

## Development

### Code Formatting

We use the uncompromising code formatter [Black](https://github.com/psf/black).

To format: `$ black {source_file_or_directory}`

### Lightning

Run `methods/lightning/main.py` with a model as argument for a default training and run. To see what models are
available, checkout the same file. For example in the root directory:

```
$ python methods/lightning/main.py unet_transformer
```

### Logging

The PyTorch Lightning framework logs automatically for tensorboard. Each run is saved in `data/lightning_logs`. You can
start Tensorboard and analyzing the versions by running following in the console at the root directory:

```
$ tensorboard --logdir data/lightning_logs
```

See [Tensorboard Get Started](https://www.tensorflow.org/tensorboard/get_started) for more information.

#### Add Hyperparameters

To add more hyperparameters to the logger, add them to the `hyper_parameter` dictionary in the LitBase constructor.

### Preprocessing

#### Divide into Four

To deal with the model using up too much memory, the images are divided into their four quadrant and the model is
trained on those smaller images. This must be accomodated when predicting, as the model learns on different zoom level
so to say. Images to predict on are divided aswell and the actual predictions are stitched back together.

To enable this, set `divide_into_four` option for the `RoadDataModule`.

See functions `_divide_into_four` and `put_back` in `ImageDataSet`