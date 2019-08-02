# This file conatins some configuration settings about the data loading
# File taken from:
# https://github.com/crowdAI/crowdai-mapping-challenge-mask-rcnn
# Note that many of these are not used in our code

import numpy as np

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations.
    NAME = None  # Override in sub-classes

    ## Path settings
    # Root directory
    ROOT_DIR = 'C:/Usman/Projects/Weak_Building_Segmentation'

    # current directory
    DIRECTORY = "./Results/Train_1"

    data_directory = "data/"
    annotation_file_template = "{}/{}/annotation{}.json"

    # Use a small subset of dataset, for quick debugging???
    SMALL_DATA = False       # IMPORTANT: must set it to False to train on full dataset


    TRAIN_IMAGES_DIRECTORY = "train/images"
    TRAIN_ANNOTATIONS_PATH = "train/annotation.json"
    TRAIN_ANNOTATIONS_SMALL_PATH = "train/annotation-small.json"

    VAL_IMAGES_DIRECTORY = "val/images"
    VAL_ANNOTATIONS_PATH = "val/annotation.json"
    VAL_ANNOTATIONS_SMALL_PATH = "val/annotation-small.json"

    ## Training Settings
    BATCH_SIZE = 16
    num_epoch = 3
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50

    # Training supervision
    SUPERVISION = 'Gaussian'    # These are the options:
                                # 'Gaussian' : prepares dense maks by converting each bounding boxes using
                                # bivariate Gaussian

                                # 'Naive': naively converts all pixels inside boundin boxes to zero and all pixels
                                # outside are labeled

                                # 'GrabCut': use OpenCV's grabcut implementation to prepare dense masks

                                # 'Full': full supervision, using true segmentation masks. This is the upper bound.


    LOSS_FN = 'Proposed_OneSided'              # 'CE' = Cross entropy
                                                # 'Proposed_OneSided' = Proposed one-sided loss

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Number of classification classes (including background)
    NUM_CLASSES = 1  # Override in sub-classes

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.

    # Input image resizing
    # Generally, use the "square" resizing mode for training and inferencing
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM is not None, then scale the small side to
    #         that size before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    IMAGE_RESIZE_MODE = "square"
    ##IMAGE_MIN_DIM = 800
    ##IMAGE_MAX_DIM = 1024

    IMAGE_MIN_DIM = 300
    IMAGE_MAX_DIM = 300

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])


    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        self.IMAGE_SHAPE = np.array(
            [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")