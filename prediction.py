import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import librosa.display
import soundfile
import os
import glob
import app
import librosa

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X, sample_rate = sound_file.read(dtype="float32"), sound_file.samplerate
        stft = np.abs(librosa.stft(X))
        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))

    return result

def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("speech-emotion-recognition-ravdess-data/Actor_*/*.wav"):
        file_name = os.path.basename(file)
        emotion = app.emotions[file_name.split("-")[2]]
        if emotion not in app.observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Load and split the dataset
x_train, x_test, y_train, y_test = load_data(test_size=0.25)
print((x_train.shape[0], x_test.shape[0]))

# Build MLP classifier
mlp_classifier = Pipeline([
    ('scaler', StandardScaler()), 
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=500,
        random_state=10
    ))
])


# Train MLP classifier
mlp_classifier.fit(x_train, y_train)

# Evaluate MLP classifier
y_pred = mlp_classifier.predict(x_test)

#Accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy=accuracy*100
print("Accuracy: ", accuracy,"%")

#Precision
# precision = precision_score(y_test, y_pred)
# precision=precision*100
# print("Precision: ", precision,"%")

cm = confusion_matrix(y_test, y_pred, labels=app.observed_emotions)

df=pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})
df.head(20)



