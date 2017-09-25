
# Generate audio as numpy array and then write wav file 

import os
import numpy as np
import scipy.io.wavfile as wf
import scipy.signal as sig
import subprocess
import math


def genSine(frequency, samplesPerSecond, durationInSeconds):
	t = np.arange(durationInSeconds * samplesPerSecond)
	sinData = np.sin(2.0 * np.pi * t * ((1.0 * frequency)/samplesPerSecond))
	return sinData

def normalizeForWav(data):
	return np.int16(data / np.max(np.abs(data)) * (maxAmplitude * (32767 - 100)))


samplesPerSecond = 44100
noiseFactor = 0.0
maxAmplitude = 0.1

specWidth = 360    # spectrogram pixels per second
specHeight = 513   # spectrogram total pixels in height


frequencyDict = {
	'C4': 261.63,
	'D4': 293.66,
	'E4': 329.63,
	'F4': 349.23,
	'G4': 392,
	'A4': 440,
	'B4': 493.88,
	'C5': 523.25
}


trainingAudioRelativeDir = 'training_audio'
trainingImagesRelativeDir = 'training_images'
testImagesRelativeDir = 'test_images'

os.makedirs(os.path.join(os.getcwd(), trainingAudioRelativeDir), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), trainingImagesRelativeDir), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), testImagesRelativeDir), exist_ok=True)


for frequencyLabel, frequency in frequencyDict.items():
	print(frequencyLabel + ' - ' + str(frequency))

	numberOfOvertones = 6

	print('Calculating fundamental')
	mySignal = genSine(frequency, samplesPerSecond, 1)

	for i in range(1, numberOfOvertones + 1):
		print('Calculating overtone ' + str(i))
		mySignal = np.add(mySignal, (1.0/(math.pow(i, 3)+1)) * genSine((i+1) * frequency, samplesPerSecond, 1))

	# introduce some noise
	if (frequencyLabel != 'silence'):
		mySignal = np.add(mySignal, noiseFactor * np.random.uniform(-1,1,samplesPerSecond))

	mySignal = normalizeForWav(mySignal)
	print(mySignal)

	filename = os.path.join(trainingAudioRelativeDir, 'sin_' + str(frequencyLabel) + '.wav')
	wf.write(filename, samplesPerSecond, mySignal)
	print('Wrote ' + filename)
	
	
	command = 'sox ' + filename + ' -n rate 6k spectrogram -m -r -X ' + str(specWidth) + ' -y ' + str(specHeight) + ' -z 80 -o ' + os.path.join(trainingImagesRelativeDir, 'sin_' + str(frequencyLabel) + '.png')
	subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
	print('Wrote training spectrogram')

