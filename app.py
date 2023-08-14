import streamlit as st
import soundfile
import matplotlib.pyplot as plt
import numpy as np
import prediction 
import os
from sklearn.metrics import ConfusionMatrixDisplay

st.title("Speech Emotion Recognition")
st.divider()
# Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Emotions to observe
observed_emotions = ['calm', 'happy','fearful', 'disgust']

dataset_folder = "speech-emotion-recognition-ravdess-data/Actor_01/"
audio_files = [file for file in os.listdir(dataset_folder) if file.endswith('.wav')]

option = st.sidebar.radio("Select Option", ("Upload File", "Select File"))

# Display upload file option
if option == "Upload File":
    soundfile = st.sidebar.file_uploader('Upload Wav sound file')
    if soundfile is not None:
        st.audio(soundfile, format="audio/wav")
        feature = prediction.extract_feature(soundfile, mfcc=True, chroma=True, mel=True)
        feature = feature.reshape(1, -1)
        predict = prediction.mlp_classifier.predict(feature)
        col1, col2= st.columns(2)
        with col1:
            st.subheader(f"The audio emotion is :")
            st.header(predict[0].capitalize())
            st.subheader(f"Accuracy of prediction:")
            st.header(f"{prediction.accuracy:.2f}%")
            st.title("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=prediction.cm, display_labels=observed_emotions)
            disp.plot(ax=ax, cmap=plt.cm.Oranges)
            st.pyplot(fig)
    
        with col2:
            st.subheader("Actual vs. Predicted Emotions")
            st.dataframe(prediction.df.head(20))
            # Display the bar graph
            correctly_predicted = np.sum(np.diag(prediction.cm))
            incorrectly_predicted = np.sum(prediction.cm) - correctly_predicted
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.bar(['Correct', 'Incorrect'], [correctly_predicted, incorrectly_predicted], color=['green', 'red'])
            ax1.set_xlabel("Prediction")
            ax1.set_ylabel("Count")
            st.pyplot(fig1)
    else:
        st.subheader("No File Selected")
        st.warning("Please upload a valid WAV audio file.",icon="🚨")

# Display select file option
elif option == "Select File":
    selected_file = st.sidebar.selectbox("Select an audio file", audio_files)
    if selected_file:
        file_path = os.path.join(dataset_folder, selected_file)
        st.audio(file_path)
        feature = prediction.extract_feature(file_path, mfcc=True, chroma=True, mel=True)
        feature = feature.reshape(1, -1)
        predict = prediction.mlp_classifier.predict(feature)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"The audio emotion is :")
            st.header(predict[0].capitalize())
            st.subheader(f"Accuracy of prediction:")
            st.header(f"{prediction.accuracy:.2f}%")
            # st.subheader(f"Precision of prediction:")
            # st.header(f"{prediction.precision:.2f}%")
            st.title("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=prediction.cm, display_labels=observed_emotions)
            disp.plot(ax=ax, cmap=plt.cm.Oranges)
            st.pyplot(fig)
        with col2:
            st.subheader("Actual vs. Predicted Emotions")
            st.dataframe(prediction.df.head(20))
            correctly_predicted = np.sum(np.diag(prediction.cm))
            incorrectly_predicted = np.sum(prediction.cm) - correctly_predicted
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.bar(['Correct', 'Incorrect'], [correctly_predicted, incorrectly_predicted], color=['green', 'red'])
            ax1.set_xlabel("Prediction")
            ax1.set_ylabel("Count")
            st.pyplot(fig1)
