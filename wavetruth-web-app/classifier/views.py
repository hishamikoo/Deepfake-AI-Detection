from django.http import JsonResponse
import traceback
import numpy as np
import pandas as pd
import librosa
from django.shortcuts import render
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from django.core.files.storage import default_storage
import os
import syllapy
import speech_recognition as sr  
from pydub import AudioSegment  
from django.conf import settings
import joblib

features_scaler_path = "Models\\scaler.pkl"
features_scaler = joblib.load(features_scaler_path)

features_mean_path = "Models\\feature_means.pkl"
features_mean = joblib.load(features_mean_path)

pca_model_path = "Models\\pca_model.pkl"
pca = joblib.load(pca_model_path)

# Load the trained model (ensure this is correctly loaded)
model_path = "Models\\transfer_learning_model.keras"
model = tf.keras.models.load_model(model_path)

LABELS = ['F_AI', 'F_REAL', 'M_AI', 'M_REAL']

def index(request):
    return render(request, 'index.html')

def convert_mp3_to_wav(mp3_path, wav_path):

    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

def extract_features(audio_path):
    try:
        # Convert the MP3 file to WAV first
        wav_path = audio_path.replace(".mp3", ".wav")
        convert_mp3_to_wav(audio_path, wav_path)

        # Load the WAV file using librosa
        y, sample_rate = librosa.load(wav_path, sr=None)

        # Extracting various features
        features = {
            "Spectral Centroid Mean": np.mean(librosa.feature.spectral_centroid(y=y, sr=sample_rate)),
            "Spectral Bandwidth Mean": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sample_rate)),
            "RMS Mean": np.mean(librosa.feature.rms(y=y)),
            "Zero-Crossing Rate Mean": np.mean(librosa.feature.zero_crossing_rate(y)),
            "Spectral Contrast Mean": np.mean(librosa.feature.spectral_contrast(y=y, sr=sample_rate)),
            "Pitch Mean": np.mean(librosa.piptrack(y=y, sr=sample_rate)[0][librosa.piptrack(y=y, sr=sample_rate)[0] > 0]),
            "Pitch Confidence Mean": np.mean(librosa.piptrack(y=y, sr=sample_rate)[1]),
            "Mel Spectrogram Mean": np.mean(librosa.feature.melspectrogram(y=y, sr=sample_rate)),
            "Mel Spectrogram Variance": np.var(librosa.feature.melspectrogram(y=y, sr=sample_rate)),
            "Energy Mean": np.mean(librosa.feature.rms(y=y)), 
        }

        # Convert audio to text & analyze speech rate
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio)
            syllable_count = sum(syllapy.count(word) for word in text.split())
            duration = librosa.get_duration(y=y, sr=sample_rate)
            features["Speech rate"] = syllable_count / duration
        except Exception as e:
            print(f"Error in speech recognition: {str(e)}")
            features["Speech rate"] = features_mean.get("Speech rate")

        # Convert features to a DataFrame with feature names
        feature_values = np.array(list(features.values())).reshape(1, -1)
        feature_df = pd.DataFrame(feature_values, columns=list(features.keys()))

        # Debug: Print feature DataFrame
        print("Feature DataFrame:", feature_df)

        # Scale the features
        scaled_features = features_scaler.transform(feature_df)

        # Debug: Print scaled features
        print("Scaled Features:", scaled_features)

        # Apply PCA transformation
        feature_values_pca = pca.transform(scaled_features)  # Pass the DataFrame with feature names

        # Debug: Print PCA-transformed features
        print("PCA-Transformed Features:", feature_values_pca)

        return feature_values_pca

    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        traceback.print_exc()
        return None


def upload_audio(request):
    try:
        if request.method == 'POST' and request.FILES.get('audio_file'):
            audio_file = request.FILES['audio_file']

            # Save the uploaded MP3 file
            file_name = 'temp_audio.mp3'
            file_path = os.path.join(settings.MEDIA_ROOT, 'audio_files', file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'wb') as f:
                f.write(audio_file.read()) 


            # Extract features
            features = extract_features(file_path)
            if features is None:
                return JsonResponse({'error': 'Error in feature extraction.'}, status=500)

            # Convert features to a DataFrame with feature names
            feature_df = pd.DataFrame(features, columns=[f"PC{i+1}" for i in range(pca.n_components_)])

            # Debug: Print extracted features
            print("Extracted Features Shape:", feature_df.shape)
            print("Extracted Features:", feature_df)

            # Predict using the model
            raw_predictions = model.predict(feature_df)

            # Debug: Print raw predictions
            print("Raw Predictions:", raw_predictions)

            # Convert predictions to probabilities
            prediction_probabilities = model.predict(feature_df)

            # Debug: Print probabilities
            print("Prediction Probabilities:", prediction_probabilities)

            # Create a dictionary of labels and their probabilities
            prediction_dict = {LABELS[i]: float(prediction_probabilities[0][i]) for i in range(len(LABELS))}

            # Return the probabilities
            return JsonResponse({'predictions': prediction_dict})
        

        return JsonResponse({'error': 'Invalid request'}, status=400)

    except Exception as e:
        print(f"Error in upload_audio view: {str(e)}")
        traceback.print_exc()
        return JsonResponse({'error': 'Error occurred during audio processing.'}, status=500)