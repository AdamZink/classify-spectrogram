# classify-spectrogram

Requirements - see windows install notes for more details
tensorflow
scipy
numpy
cv2 (opencv3)


Command to generate training data: `python generateTrainingImages.py`

Command to run the analysis program: `python classify.py`


Sample output:

--- Results ---
Slice 1 - Prediction: ['A4', 'C4', 'E4']
Slice 2 - Prediction: ['A4', 'C5', 'F4']
Slice 3 - Prediction: ['B4', 'D4', 'G4']
Slice 4 - Prediction: ['C5', 'E4', 'G4']
