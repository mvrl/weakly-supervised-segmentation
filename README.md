# Weakly Supervised Building Segmentation from Overhead Images

This is the code for our weakly supervised building segmentation paper, accepted at IGARSS 2019.

**Paper:** [Weakly Supervised Building Segmentation From Aerial Images](https://urafique.com/files/Weak_Building_Segmentation_Camera_Ready.pdf)

## Dataset

We use the disaster response dataset released as the [Mapping Challenge](https://www.crowdai.org/challenges/mapping-challenge). You should be able to get the data from this website. If you have any trouble acquiring the dataset, please contact us.

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate wbseg
```

## How to Use

All settings are stored in `wbseg/config.py`. Edit that file to set the dataset path (`ROOT_DIR`), output directory (`DIRECTORY`), supervision mode, loss function, batch size, and other training options.

**Train a model:**

```bash
python -m wbseg.train
```

This trains the U-Net model and saves the trained weights, loss curves, and metrics to the directory specified by `DIRECTORY` in `config.py`.

**Visualize results:**

```bash
python -m wbseg.visualize_trained
```

This loads the trained model and saves prediction figures to the same output directory.

### Supervision modes

Set `SUPERVISION` in `wbseg/config.py` to one of:

| Value | Description |
|---|---|
| `Gaussian` | Dense masks derived from bounding boxes using a bivariate Gaussian (default) |
| `Naive` | All pixels inside bounding boxes set to foreground |
| `GrabCut` | OpenCV GrabCut applied within each bounding box |
| `Full` | Full supervision using ground-truth segmentation masks (upper bound) |

### Loss functions

Set `LOSS_FN` in `wbseg/config.py` to one of:

| Value | Description |
|---|---|
| `Proposed_OneSided` | Proposed one-sided loss (default) |
| `CE` | Standard binary cross-entropy |

## Citation

If you find this paper or code helpful, please cite:

```
M. Usman Rafique, Nathan Jacobs, "Weakly Supervised Building Segmentation From Aerial Images",
In: IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2019.
```

## People

Please feel free to contact us with any questions or comments.

[M. Usman Rafique](https://usman-rafique.github.io/ "Usman's website")

[Nathan Jacobs](http://cs.uky.edu/~jacobs/ "Nathan's website")

## License

The code is provided for academic purposes only without any guarantees.
 