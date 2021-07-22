# Road Segmentation

_2021 | ETH Zürich | D-INFK | Computational Intelligence Lab | Course Project_

by **RO**bin Bisping, **SA**muel Bedassa and Luc**AS** Habersaat

## Idea

We combine the UNet model, together with Transformer, as done before in this [paper](https://arxiv.org/abs/2102.04306)
to segment aerial images into road and not-road. This has been done only for medical image segmentation. The strength of
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

To see what models are available, checkout the `handle_train` method in this file. If no model is
given `unet_transformer` is taken.

#### Load

To load a model, that has been trained before, use the `-load` argument with a number that indicates the version. For
instance:

```
$ python main.py -load 25
```

If no number is given, it will take the latest.
All the trained models and the respective version number can be found in `data/lightning_logs`. Train a model and it
will be logged there.

## Development

### Code Formatting

We use the uncompromising code formatter [Black](https://github.com/psf/black).

To format: `$ black {source_file_or_directory}`

### Folder Structure

Next to `main.py`, are different methods in the `methods` folder and just running them will get you started. However, as
lightning is used, it is recommended to run `methods/lightning/main.py` with parameters. See below.

Code used in multiple places is placed in `common`.

All the data is in the `data` folder:

* **training** and the **validation** data set,
* the **test images** to make the predictions at the end,
* the **submission** files with the predictions
* lightning log files (not in version control)

### Logging

The PyTorch Lightning framework logs automatically for tensorboard. Each run, or model, is saved
in `data/lightning_logs`. You can start Tensorboard and analyzing the versions by running following in the console at
the root directory:

```
$ tensorboard --logdir data/lightning_logs
```

See [Tensorboard Get Started](https://www.tensorflow.org/tensorboard/get_started) for more information.

#### Add Hyperparameters

To add more hyperparameters to the logger,
- add them to the `hyper_parameter` dictionary in the LitBase constructor It gets a bit more tricky, if it's a parameter
  for the data module.

### Preprocessing

#### Divide into Four

To deal with the model using up too much memory, the images are divided into their four quadrant and the model is
trained on those smaller images. This must be accomodated when predicting, as the model learns on different zoom level
so to say. Images to predict on are divided aswell and the actual predictions are stitched back together.

To enable this, set `divide_into_four` option for the `RoadDataModule`.

See functions `_divide_into_four` and `put_back` in `ImageDataSet`

#### Training Data Duplication
The training data is rotated by 90, 180 and 270 degrees to increase the number of training images by a factor of four.

### Postprocessing
#### Graph Cut
Instead of the threshold classifier, a graph cut is used to further improve results.
However, it takes quite some time, so it is disabled by default. To enable it, set `graph_cut` to `true` in the `write_submissions` method call in the prediction.