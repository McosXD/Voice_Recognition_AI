import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore

def load_audio_files(data_path):
    X, y = [], []
    labels = os.listdir(data_path)  
    for label in labels:
        folder_path = os.path.join(data_path, label)
        if os.path.isdir(folder_path):  
            for file in os.listdir(folder_path):  
                file_path = os.path.join(folder_path, file)
                audio, sr = librosa.load(file_path, sr=None)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13) # MFCC (Mel-Frequency Cepstral Coefficients)
                mfcc_mean = np.mean(mfcc.T, axis=0)  
                X.append(mfcc_mean)
                y.append(label)  
                
    return np.array(X), np.array(y)

data_path = "dataset_wav"  
X, y = load_audio_files(data_path)

labels = np.unique(y)
label_to_index = {label: i for i, label in enumerate(labels)}
y_indices = np.array([label_to_index[label] for label in y])

X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_indices, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acuratețe: {accuracy * 100:.2f}%")

model.save("model.keras")

def authenticate(audio_file, model, label_to_index):
    audio, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
        
    mfcc_mean_normalized = (mfcc_mean - np.mean(X, axis=0)) / np.std(X, axis=0)
        
    predictions = model.predict(mfcc_mean_normalized)
    predicted_index = np.argmax(predictions)
    for label, index in label_to_index.items():
        if index == predicted_index:
            return label
    

audio_file = "pers7.wav"
model = tf.keras.models.load_model("model.keras")
pers = authenticate(audio_file, model, label_to_index)
print(f"Utilizator identificat ca: {pers}")