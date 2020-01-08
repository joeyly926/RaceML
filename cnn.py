from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.layers.core import Activation, Flatten, Dense, Dropout, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model file")
args = vars(ap.parse_args())
 
# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the data matrix and labels list
data = []
labels = []
# 0-1
ages = []
# 0 or 1
genders = []
# 0, 1, 2, 3, 4
races = []
le = LabelEncoder()

for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	formatted_labels = imagePath.split(os.path.sep)[-1].split(".")[0]
	label = np.array(formatted_labels.split("_")[:-1], dtype=np.float)
	if len(label) != 3:
		continue
	# construct a feature vector raw pixel intensities, then update
	# the data matrix and labels list
	image = cv2.resize(image, (32, 32))
	image = img_to_array(image)

	data.append(image)
	# don't care about date and time
	# labels format: {age}_{gender}_{race}_{date&time}
	
	label[0] = label[0] / 116.0
	ages.append([label[0]])
	genders.append(label[1])
	races.append(label[2])
	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

genders = np_utils.to_categorical(genders, num_classes=2)
races = np_utils.to_categorical(races, num_classes=5)

# normalize and one hot encode this shit
data = np.array(data) / 255.0
labels = np.concatenate((ages,genders, races), axis=1)
 
# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data, races, test_size=0.25, random_state=42)
chanDim = -1
inputShape = (32, 32, 3,)
# define the architecture of the network
model = Sequential()
'''
model.add(Dense(768, input_dim=3072, init="uniform", activation="relu"))
model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
model.add(Dense(5))
model.add(Activation("softmax"))
'''
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation("softmax"))

# train the model using SGD
print("[INFO] compiling model...")
sgd = SGD(lr=0.1)
init_lr = 0.001
opt = Adam(lr=init_lr, decay=init_lr / 100)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

BS = 32
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
model.fit_generator(
	aug.flow(trainData, trainLabels, batch_size=BS),
	validation_data=(testData, testLabels),
	steps_per_epoch=len(trainData) // BS,
	epochs=100, verbose=1)

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))
 
# dump the network architecture and weights to file
print("[INFO] dumping architecture and weights to file...")
model.save(args["model"])
