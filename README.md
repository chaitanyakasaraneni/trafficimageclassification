# Traffic Image Classification

### Description:
Develop predictive models that can determine, given an image, which one of 14 classes it is. 

Traffic congestion seems to be at an all-time high. Machine Learning methods must be developed to help solve traffic problems. In this program, you will analyze features extracted from tiny traffic images depicting different traffic-related objects to determine their type as one of 14 classes, noted by integers 1-14: car, suv, small_truck, medium_truck, large_truck, pedestrian, bus, van, people, bicycle, and motorcycle, signal_green, signal_yellow, signal_red.
The object classes are heavily imbalanced. For example, the training data contains 31,775 cars but only 280 motorcycles and 197 buses. Classes in the test data are similarly distributed.

The input to our classifiers will not be the images themselves, but rather features extracted from the images. There are many models for feature extraction. A few classic examples of image features are [Histogram of Oriented Gradients (HOG)](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html) features, [Normalized Color Histogram (Hist)](https://docs.opencv.org/3.3.1/d1/db7/tutorial_py_histogram_begins.html) features, [Local Binary Pattern (LBP)](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html) features, Color gradient (RGB) features, [Depth of Field (DF)](https://en.wikipedia.org/wiki/Depth_of_field) features, etc.

### Data Description:
The training dataset consists of 100,000 records and the test dataset also consists of 100,000 records. The training class labels are also given.
Note that the dataset file is very large (474 MB, expands to 1050 MB). 
The dataset file contains two dataset directories: traffic, and traffic-small. In each directory, you will find train and test sub-directories with images numbered 1 to 100,000 (e.g., 000001.jpg) for the traffic dataset and 1 to 4209 for the traffic-small dataset. The train and
test sets contain the same number of images. The image ID for the ith image corresponds to the label on the ith line of the train.labels file found in the main dataset directory. The traffic-small dataset contains a test.labels file, but the traffic dataset does not.
Our task is to predict those labels for the images in the test set and create a .txt or .ddat file containing those labels.

### Requirements
You need to install [tensorflow](https://anaconda.org/conda-forge/tensorflow) and [keras](https://anaconda.org/conda-forge/keras) to run this program which you can do by using pip.
```
pip install keras
pip install tensorflow
```
### My Implementation
Here I used the color gradient features (grayscale) and tensorflow's inbuilt image to array conversion for feature extraction. 
```
img = image.load_img('traffic/traffic/train/'+train['id'][i],
                    target_size = (28,28,1), color_mode = "grayscale") ##feature selection
img = image.img_to_array(img) ##image to array conversion
```
Then, split the data into training and testiing to 80% and 20% respectively using:
```
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
```
After that, define 2 layered Convolution Neural Network (CNN) for training and classifying the images.
```
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='softmax'))
```
### References
- [Program Reference](https://www.analyticsvidhya.com/blog/2019/01/build-image-classification-model-10-minutes/)
- [Adam Optimizer Algorithm](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
- [ReLU, Softmax Activation Functions](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
- [Convolutional Neural Networks](https://keras.io/layers/convolutional/)
- [Dense, Dropout functions](https://keras.io/layers/core/)
