# This file loads a trained model and saves figures in the same folder.

# Author M. Usman Rafique

## Imports
import os

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from dataset import MappingChallengeDataset
from keras.models import load_model

from config import Config as CFG

from dataset import data_generator_modified




## Prepare datasets
# Load validation dataset
dataset_val = MappingChallengeDataset()
val_coco = dataset_val.load_dataset(dataset_dir=os.path.join(CFG.ROOT_DIR, "val"), load_small=True, return_coco=True)
dataset_val.prepare()

# Prepare generators
val_generator = data_generator_modified(dataset_val, CFG, shuffle=True,
                                       batch_size=CFG.BATCH_SIZE)


##################### NEW automation
directory = CFG.DIRECTORY

if not os.path.exists(directory):
    raise ValueError('The folder does not exist')


# Load trained model
model = load_model(os.path.join(directory,'trained_model_full.h5'))

# coco = COCO(VAL_ANNOTATIONS_PATH) # Uncomment if you want to access the full dataset
coco = COCO(os.path.join(CFG.ROOT_DIR, CFG.VAL_ANNOTATIONS_SMALL_PATH)) # small ataset, faster

## Visualizing and saving figures

my_th = 0.5  # threshold used to generate binary masks

for index, (batch_images, outputs) in enumerate(val_generator):
    prediction = model.predict_on_batch(batch_images)

    # for every prediction in the batch
    for k in range(batch_images[0].shape[0]):
        # image_show = batch_images[0][k, :, :, :]  # Mean is subtracted from this image, not good for displaying
        # Manually load the image, using image ID. This is the reason of keeping image IDs ...
        current_id = int(outputs[2][k, :])
        image_show = dataset_val.load_image(current_id)

        pred_segm = prediction[0][k, :, :, 0]
        pred_segm_bin = pred_segm > my_th

        gauss_mask = outputs[0][k, :, :, 0]  # Gaussian mask
        gt_mask = outputs[1][k, :, :, 0]  # segmentation mask

        plt.figure()
        plt.subplot(1, 5, 1)
        plt.imshow(image_show)
        plt.title('Input')
        plt.axis('off')
        plt.subplot(1, 5, 2)
        plt.imshow(gauss_mask)
        plt.title('Gaussian')
        plt.axis('off')
        plt.subplot(1, 5, 3)
        plt.imshow(pred_segm_bin)
        plt.title('Threshd O/P')
        plt.axis('off')
        plt.subplot(1, 5, 4)
        plt.imshow(pred_segm)
        plt.title('Raw O/P')
        plt.axis('off')
        plt.subplot(1, 5, 5)
        plt.imshow(gt_mask)
        plt.title('GT Seg')
        plt.axis('off')
        #plt.show()
        #plt.pause(1)

        fname1 = str('result'+str(index)+ '_' +str(k)+'.png')
        plt.savefig(os.path.join(directory, fname1))
        plt.clf()

    if index >=  2:       # Save batches = 3x16 = 48 figures
        break

print('Saved images in the same directory...')