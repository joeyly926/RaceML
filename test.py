from keras.models import load_model
from keras.preprocessing.image import img_to_array
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
 
def image_to_feature_vector(image, size=(32, 32)):
	# resize and flatten iamge
	return cv2.resize(image, size).flatten()

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model file")
ap.add_argument("-t", "--test-images", required=True,
	help="path to the directory of testing images")
ap.add_argument("-b", "--batch-size", type=int, default=32,
	help="size of mini-batches passed to network")
args = vars(ap.parse_args())


# class labels for races
CLASSES = ["White", "Black", "Asian", "Indian", "Others"]
 
# load the network
print("[INFO] loading network architecture and weights...")
model = load_model(args["model"])
print("[INFO] testing on images in {}".format(args["test_images"]))


# loop over over test images
for imagePath in paths.list_images(args["test_images"]):
	# process image
	print("[INFO] classifying {}".format(imagePath[imagePath.rfind("/") + 1:]))
	image = cv2.imread(imagePath)
	features = cv2.resize(image, (32, 32))
	features = img_to_array(features) / 255.0
	features = np.expand_dims(features, axis=0)

	# predict race from image
	probs = model.predict(features)[0][-51:]
	print(probs)
	prediction = probs.argmax(axis=0)
	
	# display results
	label = "{}: {:.2f}%".format(CLASSES[prediction],
		probs[prediction] * 100)
	cv2.putText(image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 255, 0), 3)
	cv2.imshow("Image", image)
	cv2.waitKey(0)