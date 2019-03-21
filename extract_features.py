# import packages
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from pipeline.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
    help = "path to input dataset")
ap.add_argument("-o", "--output", required = True,
    help = "path to HDF5 file")
ap.add_argument("-b", "--batch_size", type = int, default = 32,
    help = "batch size of images to be passed through network")
ap.add_argument("-s", "--buffer_size", type = int, default = 1000,
    help = "size of feature extraction buffer")
args = vars(ap.parse_args())

# store the batch size in a variable
bs = args["batch_size"]

# grab the list of images and shuffle them
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

# extract the class labels from the image paths then encode the labels
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load the VGG16 network
print("[INFO] loading network...")
model = VGG16(weights = "imagenet", include_top = False)

# initialize the HDF5 dataset writer
# then store the class label names in the dataset
dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7), args["output"],
    dataKey = "features", bufSize = args["buffer_size"])
dataset.storeClassLabels(le.classes_)

# initialize the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
    progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval = len(imagePaths), widgets = widgets).start()

# loop over images in patches
for i in np.arange(0, len(imagePaths), bs):
    # extract the batch of images and labels, then initialize the list of
    # actual images that will be passed through the network for feature extraction
    batchPaths = imagePaths[i : i + bs]
    batchLabels = labels[i : i + bs]
    batchImages = []

    # loop over the images and labels in the current batch
    for (j, imagePath) in enumerate(batchPaths):
        # load the input image using the Keras util function
        # and ensure to resize the image to 224x224
        image = load_img(imagePath, target_size = (224, 224))
        image = img_to_array(image)

        # preprocess the image by
        # (1) expanding the dimensions
        # (2) subtracting the mean RGB pixel intensity from the ImageNet dataset
        image = np.expand_dims(image, axis = 0)
        image = imagenet_utils.preprocess_input(image)

        # add the image to the batch
        batchImages.append(image)

    # pass the images through the network and use the outputs as actual features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size = bs)

    # reshape the features so that each image is represented by
    # a flattened feature vector of the output
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # add the features and labels to HDF5 dataset
    dataset.add(features, batchLabels)
    pbar.update(i)

# close the dataset
dataset.close()
pbar.finish()
