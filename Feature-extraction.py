import librosa
import numpy as np
import os
import pandas as pd
import syllapy
import speech_recognition as speech
from pydub import AudioSegment


# Path to the folder containing audio clips
audio_folder_path = 'Audio_clips_FolderPath'

# List to store feature data for all audio clips
audio_features_list = []

def mp3_to_wav(mp3_path, wav_path):
    # Load the MP3 file
    audio = AudioSegment.from_mp3(mp3_path)
    # Export as WAV
    audio.export(wav_path, format="wav")

for filename in os.listdir(audio_folder_path):
    if filename.endswith('.mp3'):  
        audio_path = os.path.join(audio_folder_path, filename)
        
        file_size_kb = os.path.getsize(audio_path) / 1024  # Convert bytes to KB
        
        y, sr = librosa.load(audio_path, sr=None)
        
        # -------Extract features----------
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast)
        
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        
        pitch, pitch_confidence = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitch[pitch > 0])  # Filter to include only non-zero pitches
        pitch_confidence_mean = np.mean(pitch_confidence)

        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spectrogram_mean = np.mean(mel_spectrogram)
        mel_spectrogram_var = np.var(mel_spectrogram)

        energy = librosa.feature.rms(y=y)  # calculates energy (RMS) over time
        energy_mean = energy.mean()


        # Path to your MP3 file
        mp3_file = audio_path
        wav_file = "audio_clip.wav"

        # Convert MP3 to WAV
        mp3_to_wav(mp3_file, wav_file)

        recognizer = speech.Recognizer()
        # Load the audio file using SpeechRecognition
        with speech.AudioFile(wav_file) as source:
            audio = recognizer.record(source)  # Read the entire file

        try:
            transcription = recognizer.recognize_google(audio)  # Using Google Web Speech API
            
            words = transcription.split()
            syllable_count = sum(syllapy.count(word) for word in words)  # Count syllables in each word
            
            audio_duration = AudioSegment.from_wav(wav_file).duration_seconds  # Duration in seconds
            speech_rate = syllable_count / audio_duration  # Syllables per second
            
        except speech.UnknownValueError:
            speech_rate = 0
        except speech.RequestError as e:
            speech_rate = 0


        # Compile all features into a single row
        feature_row = {
            'Filename': filename,
            'File Size (KB)': file_size_kb,
            'Spectral Centroid Mean': spectral_centroid_mean,
            'Spectral Bandwidth Mean': spectral_bandwidth_mean,
            'RMS Mean': rms_mean,
            'Zero-Crossing Rate Mean': zero_crossing_rate_mean,
            'Spectral Contrast Mean': spectral_contrast_mean,
            'Pitch Mean': pitch_mean,
            'Pitch Confidence Mean': pitch_confidence_mean,
            'Mel Spectrogram Mean': mel_spectrogram_mean,
            'Mel Spectrogram Variance': mel_spectrogram_var,
            'Energy Mean': energy_mean,
            'Speech rate': speech_rate
        }
        
        audio_features_list.append(feature_row)

# Convert to a DataFrame
df = pd.DataFrame(audio_features_list)

# Save to Excel
output_path = 'cr_dataset.xlsx'
df.to_excel(output_path, index=False)

print(f"Features extracted and saved to {output_path}")
