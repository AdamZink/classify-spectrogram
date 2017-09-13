import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Above line is to ignore informational warnings
# https://github.com/tensorflow/tensorflow/issues/7778

import tensorflow as tf
import numpy as np
import cv2

# SOX command to generate spectrogram:
# sox session.wav -n remix 2 rate 6k spectrogram -m -r -X <specWidth> -y <specHeight> -z 80 -o short_song.png

testLength = 4  # number of seconds in test spectrogram
specWidth = 360
specHeight = 513
splitSize = 360

trainingCount = 8

# 0 will keep only minimum value (best match)
# increasing threshold will keep values increasingly higher than minimum (less perfect matches)
thresholdFromMinimum = 0.25


if (specWidth % splitSize != 0):
	print('Error - splitSize (' + str(splitSize) + ') must evenly divide specWidth(' + str(specWidth) + ')')
	exit()

	# import spectrogram and format as numpy array
def get1dSpectrogramSlices(dir, filename, imgWidth, numSlices=1):
	img = cv2.imread(filename, 0) / 255.0
	slicesArray = np.split(img, int(imgWidth/splitSize), axis=1)
	resultArray = np.empty(shape=(0, splitSize*specHeight))
	sliceCount = 0
	for slice in slicesArray:
		slice = slice.flatten()
		slice = np.expand_dims(slice, axis=0)
		resultArray = np.append(resultArray, slice, axis=0)
		sliceCount += 1
		if (sliceCount >= numSlices):
			break
	#print('get1dSpectrogramSlices return shape - ' + str(resultArray.shape))
	return resultArray
	
def getImageData(dir, imgWidth, numSlices=1):
	priorDir = os.getcwd()
	os.chdir(dir)
	data = np.empty(shape=(0, splitSize*specHeight))
	labels = np.empty(shape=(0, 1))
	for filename in os.listdir(dir):
		imgSliceArray = get1dSpectrogramSlices(dir, filename, imgWidth, numSlices)
		for img in imgSliceArray:
			img = np.expand_dims(img, axis=0)
			data = np.append(data, img, axis=0)
			label = np.expand_dims(np.array([filename.replace('sin_', '').replace('.png', '')]).flatten(), axis=0)
			labels = np.append(labels, label, axis=0)
	os.chdir(priorDir)
	#print('getImageData return shape - data: ' + str(data.shape) + ', labels: ' + str(labels.shape))
	return (data, labels)


# Get sample of each training spectrogram equal to width of 1 slice
training_notes, training_labels = getImageData(os.path.join(os.getcwd(), 'training_images'), specWidth, 1)
print(str(training_notes.shape) + ' -> ' + str(training_notes))
print(str(training_labels.shape) + ' -> ' + str(training_labels))


# Get samples of each test spectrogram equal to number of slices fitting evenly in specWidth
test_notes, test_labels = getImageData(os.path.join(os.getcwd(), 'test_images'), testLength*specWidth, testLength*int(specWidth/splitSize))
#print(str(test_notes.shape) + ' -> ' + str(test_notes))
#print(str(test_labels.shape) + ' -> ' + str(test_labels))


#print('\n--- Graph Setup ---')

training_notes_pl = tf.placeholder(tf.float32, shape=(trainingCount, splitSize*specHeight))
test_note_pl = tf.placeholder(tf.float32, shape=(1, splitSize*specHeight))

myZeros = tf.zeros(shape=(1, splitSize*specHeight))

test_masked = tf.zeros(shape=(0, splitSize*specHeight))


for myTrainingTensor in tf.split(training_notes_pl, trainingCount):
	trainingIsZero = tf.equal(myTrainingTensor, myZeros)
	one_test_masked = tf.where(trainingIsZero, myZeros, test_note_pl)
	test_masked = tf.concat([test_masked, one_test_masked], axis=0)

#print(test_masked.shape)
#print(training_notes_pl.shape)
l1_distance_modified = tf.abs(tf.add(training_notes_pl, tf.negative(test_masked)))

distance = tf.reduce_sum(l1_distance_modified, axis=1)

max = tf.arg_max(distance, 0)
min = tf.arg_min(distance, 0)
skewed_average = distance[min] + ((distance[max] - distance[min]) / tf.constant(1.0 / thresholdFromMinimum, dtype=tf.float32))


print('\n--- Results ---')

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	
	for i in range(len(test_notes)):
		
		distanceList = sess.run(distance, 
			feed_dict={
				training_notes_pl: training_notes, 
				test_note_pl: test_notes[i, :].reshape((1, splitSize*specHeight))
			})
			
		threshold = sess.run(skewed_average, 
			feed_dict={
				distance: distanceList
			})
			
		#print(str(distanceList) + ' - less than ' + str(threshold) + '?')
		
		labelIndex = 0
		resultLabels = []
		for dist in distanceList:
			if (dist < threshold):
				resultLabels.append(training_labels[labelIndex].item())
			labelIndex += 1
		
		print('Slice ' + str(i+1) + ' - Prediction: ' + str(resultLabels))
		
