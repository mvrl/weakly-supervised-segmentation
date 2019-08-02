# Weakly Supervused Building Segmentation from Overhead Images
This is the repo for our wekaly supervised building segmentation, accepted at IGARSS 2019.

## Dataset
We use the disaster response dataset, released as the [mapping challenege:](https://www.crowdai.org/challenges/mapping-challenge). You should be able to get the data form this website. If you have any trouble in acquiring the dataset, please contact us.

## How to use this code
There are several files, comments at the top of each file explain the purpose.

All the settings are stored in `config.py`. It is expected that you will forst train a model and then you can run visualization code.

To train a model, specify dataset path, training settings (level of supervision, loss function, batch size etc) in `config.py`. You will also specify a directory in which the trained model will be saved. Once the training finishes, a log file and loss curves will be saved in that folder.

After training, you can run the `visualize_trained.py` file, which will load the trained model and save some visual results in that folder.

## Citation
If you find this paper or code helpful, please cite this paper:

M. Usman Rafique, Nathan Jacobs, "Weakly Supervised Building Segmentation From Aerial Images",  In: IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2019. 

## People
Please feel free to contact us for any question or comment.

[M. Usman Rafique](https://usman-rafique.github.io/ "Usman's website")

[Nathan Jacobs](http://cs.uky.edu/~jacobs/ "Nathan's website")

## Permission
The code is provided for academic purposes only without any guarantees. 