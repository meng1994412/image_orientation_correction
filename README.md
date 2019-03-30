# Image Orientation Correction
## Objectives
Apply transfer learning to automatically detect and correct orientation of an image.
* Modified `Indoor CVPR` dataset images by rotating the images into 0, 90, 180, 270 degrees separately to build a new dataset.
* Extracted features via `VGG16` network pre-trained on ImageNet and save features into `hdf` file.
* Trained a logistic regression classifier built on top of the `VGG16` to correct orientation classifier and evaluate the model.
* Defined an end-to-end pipeline so that we can input an image and its orientation will be corrected.

## Packages Used
* Python 3.6
* [OpenCV](https://docs.opencv.org/3.4.4/) 4.0.0
* [keras](https://keras.io/) 2.1.0
* [Tensorflow](https://www.tensorflow.org/install/) 1.13.0
* [cuda toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0
* [cuDNN](https://developer.nvidia.com/cudnn) 7.4.2
* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/scipylib/index.html)

## Approaches
The dataset used in the project is `Indoor CVPR` ([reference](http://web.mit.edu/torralba/www/indoor.html)). The dataset contains 15620 total images, which has 67 indoor room/scene categories, including homes, offices, public spaces, stores, and etc.

### Build dataset
The `create_dataset.py` ([check here](https://github.com/meng1994412/image_orientation_correction/blob/master/create_dataset.py)) is responsible for randomly (uniformly) rotating images either by 0 (no change), 90 degrees, 180 (flipped vertically) degrees, or 270 degrees. Thus, there are four categories, each having about 3600 images per angle.

### Extract features
The `extract_features.py` ([check here](https://github.com/meng1994412/image_orientation_correction/blob/master/extract_features.py)) is responsible for extracting features via `VGG16` network pre-trained on ImageNet.

Here is a helper function:

The `hdf5datasetwriter.py` ([check here](https://github.com/meng1994412/image_orientation_correction/blob/master/pipeline/io/hdf5datasetwriter.py)) under `pipeline/io/` directory, defines a class that help to write raw images or features into `HDF5` dataset.

### Train and evaluate the logistic regression classifier
The `train_model.py` ([check here](https://github.com/meng1994412/image_orientation_correction/blob/master/train_model.py)) is responsible for training and evaluating the logistic regression classifier, building on top the `VGG16` network.

### End-to-end orientation correction pipeline
The `orient_images.py` ([check here](https://github.com/meng1994412/image_orientation_correction/blob/master/orient_images.py)) builds an end-to-end orientation correction pipeline, which we can input an image and its orientation will be corrected accordingly.

## Results
### Train and evaluate the logistic regression classifier
The Figure 1 shows the evaluation of the logistic regression classifier.

<img src="https://github.com/meng1994412/image_orientation_correction/blob/master/outputs/evaluation.png" width="400">

Figure 1: Evaluation of the logistic regression classifier.

### End-to-end orientation correction pipeline
Figure 2 to 5 demonstrate some sample outputs for the original and corrected images.

<img src="https://github.com/meng1994412/image_orientation_correction/blob/master/outputs/sample_output_1.png" width="500">

Figure 2: Sample output #1.

<img src="https://github.com/meng1994412/image_orientation_correction/blob/master/outputs/sample_output_2.png" width="500">

Figure 3: Sample output #2.

<img src="https://github.com/meng1994412/image_orientation_correction/blob/master/outputs/sample_output_3.png" width="500">

Figure 4: Sample output #3.

<img src="https://github.com/meng1994412/image_orientation_correction/blob/master/outputs/sample_output_4.png" width="500">

Figure 4: Sample output #4.
