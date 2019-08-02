# Different utility functions

# Most of dataloading is adopted from the baseline training repo of the dataset, available at:
# https://github.com/crowdAI/crowdai-mapping-challenge-mask-rcnn
# However, please note that significant changes have been made to the code.

import utils
import numpy as np
import random


from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from pycocotools import mask as cocomask
from scipy.stats import multivariate_normal

import os
import cv2 as cv    # Needed for grabCut
import logging
from config import Config as CFG

class MappingChallengeDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, load_small=False, return_coco=True):
        """ Loads dataset released for the crowdAI Mapping Challenge(https://www.crowdai.org/challenges/mapping-challenge)
            Params:
                - dataset_dir : root directory of the dataset (can point to the train/val folder)
                - load_small : Boolean value which signals if the annotations for all the images need to be loaded into the memory,
                               or if only a small subset of the same should be loaded into memory
        """
        self.load_small = load_small
        if self.load_small:
            annotation_path = os.path.join(dataset_dir, "annotation-small.json")
        else:
            annotation_path = os.path.join(dataset_dir, "annotation.json")

        image_dir = os.path.join(dataset_dir, "images")
        print("Annotation Path ", annotation_path)
        print("Image Dir ", image_dir)
        assert os.path.exists(annotation_path) and os.path.exists(image_dir)

        self.coco = COCO(annotation_path)
        self.image_dir = image_dir

        # Load all classes (Only Building in this version)
        classIds = self.coco.getCatIds()

        # Load all images
        image_ids = list(self.coco.imgs.keys())

        # register classes
        for _class_id in classIds:
            self.add_class("crowdai-mapping-challenge", _class_id, self.coco.loadCats(_class_id)[0]["name"])

        # Register Images
        for _img_id in image_ids:
            assert(os.path.exists(os.path.join(image_dir, self.coco.imgs[_img_id]['file_name'])))
            self.add_image(
                "crowdai-mapping-challenge", image_id=_img_id,
                path=os.path.join(image_dir, self.coco.imgs[_img_id]['file_name']),
                width=self.coco.imgs[_img_id]["width"],
                height=self.coco.imgs[_img_id]["height"],
                annotations=self.coco.loadAnns(self.coco.getAnnIds(
                                            imgIds=[_img_id],
                                            catIds=classIds,
                                            iscrowd=None)))

        if return_coco:
            return self.coco

    # Loads true segmentation mask. Not used for training.
    def load_mask(self, image_id):
        """ Loads instance mask for a given image
              This function converts mask from the coco format to a
              a bitmap [height, width, instance]
            Params:
                - image_id : reference id for a given image

            Returns:
                masks : A bool array of shape [height, width, instances] with
                    one mask per instance
        """

        image_info = self.image_info[image_id]
        assert image_info["source"] == "crowdai-mapping-challenge"

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "crowdai-mapping-challenge.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation,  image_info["height"],
                                                image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue

                # Ignore the notion of "is_crowd" as specified in the coco format
                # as we donot have the said annotation in the current version of the dataset
                instance_masks.append(m)
                class_ids.append(class_id)
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(MappingChallengeDataset, self).load_mask(image_id)


    ## Naive masks, by setting everything inside the bounding boxes to be 1, everything else is 0.
    def load_mask_naive(self, image_id):
        """ Load mask from bounding boxes only in 
              a bitmap [height, width, 1] i.e. there will be only one mask
            Params:
                - image_id : reference id for a given image

            Returns:
                masks : A bool array of shape [height, width, instances] with
                    one mask per instance
                class_ids : a 1D array of classIds of the corresponding instance masks
                    (In this version of the challenge it will be of shape [instances] and always be filled with the class-id of the "Building" class.)
        """

        image_info = self.image_info[image_id]
        assert image_info["source"] == "crowdai-mapping-challenge"

        annotations = self.image_info[image_id]["annotations"]
        test_mask = np.zeros((image_info["height"], image_info["width"]))
        test_mask_padded = np.zeros((image_info["height"] + 4, image_info["width"] + 4))
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "crowdai-mapping-challenge.{}".format(annotation['category_id']))
            if class_id:
                rle = cocomask.frPyObjects(annotation['segmentation'] , image_info["height"], image_info["width"])
                m = cocomask.decode(rle)
                test = utils.extract_bboxes(m)  # Extract bounding box

                test_mask[int(test[0][0]):int(test[0][2]):1, int(test[0][1]):int(test[0][3]):1] = 1;

        # padded mask
        test_mask_padded[2:302, 2:302] = test_mask

        # return the final mask
        return test_mask_padded

    ## Code for GrabCut. This method trains a CNN with GrabCut supervision

    # Note: this result is not presented in the paper.
    # In the paper, we show GrabCut + Oracel: even at test time, bounding boxes are provided to the Grabcut method.

    def load_mask_GrabCut(self, image_id):#, image):
        image = self.load_image(image_id)
        image_info = self.image_info[image_id]
        assert image_info["source"] == "crowdai-mapping-challenge"
        annotations = self.image_info[image_id]["annotations"]

        test_mask_cut = np.zeros((image_info["height"]+4, image_info["width"]+4))  # initialize
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "crowdai-mapping-challenge.{}".format(annotation['category_id']))
            if class_id:
                rle = cocomask.frPyObjects(annotation['segmentation'], image_info["height"], image_info["width"])
                m = cocomask.decode(rle)
                test = utils.extract_bboxes(m)

                bgdModel = np.zeros((1, 65), np.float64)  # needed for internal working - DON'T Change
                fgdModel = np.zeros((1, 65), np.float64)

                grab_mask_in = np.zeros(image.shape[:2], np.uint8)
                grab_mask_out = np.zeros(image.shape[:2], np.uint8)

                # GrabCUT
                # rect = [x, y, w, h]
                h = int(test[0][2]) - int(test[0][0])
                if (h==300):
                    h = 299
                w = int(test[0][3]) - int(test[0][1])
                if (w==300):
                    w = 299
                rect = (int(test[0][1]), int(test[0][0]), w, h)

                if (np.linalg.norm(rect)==0):    # empty
                    continue

                if (np.linalg.norm(rect)==0 or (h<3) or (w<3)):    # too small bounding boxes crash OpenCV.GrabCut
                    grab_mask_out[int(test[0][0]):int(test[0][2]):1, int(test[0][1]):int(test[0][3]):1] = 1;    #just set the whole small window as the FG
                    continue

                cv.grabCut(image, grab_mask_in, rect, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_RECT)
                grab_mask_out = np.where((grab_mask_in == 2) | (grab_mask_in == 0), 0, 1).astype(np.float64)

                test_mask_cut[2:302, 2:302] += grab_mask_out

        del bgdModel, fgdModel, grab_mask_in, image, grab_mask_out
        # make sure the max value is one
        test_mask_cut = np.clip(test_mask_cut, np.min(test_mask_cut), 1)
        # return the final mask
        return test_mask_cut

    # Function that converts bounding boxes to Gaussian masks
    def load_mask_gaussian(self, image_id):
        """ Load masks but with Gaussian distribution instead of a binary mask
            Params:
                - image_id : reference id for a given image

            Returns:
                masks : A bool array of shape [height, width, instances] with
                    one mask per instance
        """

        image_info = self.image_info[image_id]
        assert image_info["source"] == "crowdai-mapping-challenge"

        annotations = self.image_info[image_id]["annotations"]

        test_mask = np.zeros((image_info["height"], image_info["width"]))
        #gaussian_mask = np.zeros((image_info["height"], image_info["width"]))
        test_mask_padded = np.zeros((image_info["height"]+4, image_info["width"]+4))
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "crowdai-mapping-challenge.{}".format(annotation['category_id']))
            if class_id:
                rle = cocomask.frPyObjects(annotation['segmentation'] , image_info["height"], image_info["width"])
                m = cocomask.decode(rle)

                test = utils.extract_bboxes(m)
                wy = int(test[0][2]) - int(test[0][0])
                wx = int(test[0][3]) - int(test[0][1])

                test_mask[int(test[0][0]):int(test[0][2]):1, int(test[0][1]):int(test[0][3]):1] = 1
                if wy < 3 or wx < 3:        # If the bounding box is very small, return a binary mask
                    # Binary mask
                    test_mask[int(test[0][0]):int(test[0][2]):1, int(test[0][1]):int(test[0][3]):1] = 1
                else:
                    # Gaussian mask
                    coords = np.mgrid[int(test[0][0]):int(test[0][2]):1, int(test[0][1]):int(test[0][3]):1].reshape(2, -1).T

                    my_mean = [0.5 * test[0][0] + 0.5 * test[0][2], 0.5 * test[0][1] + 0.5 * test[0][3]]

                    sigx = (wx*wx) /4.0
                    sigy = (wy*wy) /4.0

                    my_cov = [[sigy, 0], [0, sigx]]

                    gauss_2dd = multivariate_normal(my_mean, my_cov)
                    vall = gauss_2dd.pdf([coords]).reshape(wy, wx)

                    vall = vall / np.max(vall)  # normalize to have max value of 1

                    test_mask[int(test[0][0]):int(test[0][2]):1, int(test[0][1]):int(test[0][3]):1] = vall

        # return the final, padded mask
        test_mask_padded[2:302, 2:302] = test_mask[:,:]

        return test_mask_padded

    def image_reference(self, image_id):
        """Return a reference for a particular image

            Ideally you this function is supposed to return a URL
            but in this case, we will simply return the image_id
        """
        return "crowdai-mapping-challenge::{}".format(image_id)
    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

## Helper functions for data loading
def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def load_image_gt(dataset, config, image_id, augment=False, augmentation=None):
    """Load and return ground truth data for an image (image, mask, bounding boxes).
    augment: (Depricated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    mask: [height, width, instance_count]. The height and width are those
        of the image
    gaussian_mask: masks from the bounding boxes. Dense masks are made by using bivariate Gaussian distribution
    """
    # Load image and mask

    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)

    # Decide the level of supervision, defined in config.py
    if CFG.SUPERVISION =='Gaussian':
        weak_mask = dataset.load_mask_gaussian(image_id)  # Gaussian masks

    elif CFG.SUPERVISION =='Naive': # Naive masks
        weak_mask = dataset.load_mask_naive(image_id)

    elif CFG.SUPERVISION =='GrabCut': # Grabcut
        weak_mask = dataset.load_mask_GrabCut(image_id)

    elif  CFG.SUPERVISION =='Full':     # Fully supervised (with true segmentation masks)
        weak_mask = mask

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        logging.warning("'augment' is depricated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            weak_mask = np.fliplr(weak_mask)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmentors that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return (augmenter.__class__.__name__ in MASK_AUGMENTERS)

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        weak_mask_shape = weak_mask.shape    ##my addition

        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))

        weak_mask = det.augment_image(weak_mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))

        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        assert weak_mask.shape == weak_mask_shape, "Augmentation shouldn't change Usman's mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

        weak_mask = weak_mask.astype(np.bool)

    return image, mask, weak_mask


## Data generator
def data_generator_modified(dataset, config, shuffle=True, augment=False, augmentation=None,
                   random_rois=0, batch_size=1, detection_targets=False):
    """A generator that returns images, bounding box-based masks for training, segmentation masks for visualization and
     evaluation, object ID so that images and other annotations can be loaded
    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs.
    """

    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, gt_masks, gaussian_mask = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              augmentation=augmentation)

            # Skip images that have no instances.
            if gt_masks.shape[0]==0:        # if the true mask is empty (it is possible!), skip that image
                continue

            # Init batch arrays
            if b == 0:
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)

                # Images
                batch_images_padded = np.zeros(
                    (batch_size, gt_masks.shape[1]+4, gt_masks.shape[1]+4,3), dtype=np.float32)

                # batch image ids
                batch_image_ids = np.zeros((batch_size, 1))

                # True segmentation mask, not used for training
                batch_GT_mask_padded = np.zeros((batch_size, gt_masks.shape[1]+4, gt_masks.shape[1]+4, 1), dtype=np.bool)

                # Gaussian masks, used for training
                batch_gaussian_mask = np.zeros((batch_size, gaussian_mask.shape[1], gaussian_mask.shape[1], 1))

            batch_images[b] = mold_image(image.astype(np.float32), config)    ## subtract the mean

            # Gaussian masks, in a batch
            batch_gaussian_mask[b, :, :, 0] = gaussian_mask

            # Combining all instances of buildings into a single segmentaition mask
            for i in range(gt_masks.shape[2]):
                incoming = gt_masks[:, :, i]
                incoming = incoming.astype(np.bool)
                batch_GT_mask_padded[b, 2:302, 2:302, 0] += incoming


            # Update batch image IDs
            batch_image_ids[b, 0] = image_id

            # padded images
            batch_images_padded[b,2:302, 2:302,:] = batch_images[b, :, :]

            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images_padded ]

                outputs = [batch_gaussian_mask, batch_GT_mask_padded, batch_image_ids]  #batch_GT_mask_oneLayer

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise

