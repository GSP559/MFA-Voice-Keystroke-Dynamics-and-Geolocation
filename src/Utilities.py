# utils.py
import numpy as np
import sounddevice as sd
import librosa
import time
import soundfile as sf
import random
import pickle
from scipy.spatial.distance import euclidean

def extract_features_with_pitch_and_tone(file_path):
#   Extract various features from voice file        
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pitch, _ = librosa.piptrack(y=audio, sr=sample_rate)
    pitch_mean = np.mean(pitch, axis=1)
    mfccs_processed = np.mean(mfccs.T, axis=0)

#   Extracting tone-related features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)

#   Mean values of tone-related features
    spectral_centroids_mean = np.mean(spectral_centroids)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)

#   Combining all features
    combined_features = np.hstack((mfccs_processed, pitch_mean, spectral_centroids_mean, spectral_bandwidth_mean, zero_crossing_rate_mean))
    return combined_features

#POSSIBLY EDIT THIS FUNCTION TO RETURN SOMETHING ELSE IF TRUE SO WE CAN USE IN DECISION FUNCTION
def Comparison(file1, file2):
#   Compare features of two voice models 
    distance = euclidean(file1, file2)

    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([file1], [file2])[0][0]
    print("This is the similarity score using cosine: " + str(similarity))
    correlation = np.corrcoef(file1, file2)[0, 1]

    print("Correlation coeff: " + str(correlation))
    print("Euc Distance: " + str(distance))
    avg = correlation + similarity
    avg = (avg/2)
    print("Average: " + str(avg))
#    Determine if they are the same person based on a threshold
    threshold = 900  # Can be adjusted
    if distance < threshold:
        if avg > .95:
            return avg
    else:
        return False
    
def record_voice(filename, duration=15, sr=22050, device=None):
#   Record a users voice and save to filename 
    voice_sentences = [
        "The quick brown fox jumps over the lazy dog",
        "Bananas are great",
        "I love coffee",
        "Rainy days are gloomy",
        "Technology is amazing"
    ]
    sentence = voice_sentences
    print(f"Please say the following sentences: {sentence}")
    
    audio_data = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='int16', device=device)
    sd.wait()
    sf.write(filename, audio_data, sr)

def record_typing():
#   Record a users typing speed 
    typing_sentences = [
        "Every moment is a fresh beginning",
        "Change the world by being yourself",
        "Welcome to Fresno State Computer Science",
        "Never regret anything that made you smile",
        "Die with memories, not dreams"
    ]

    print(f"Please type the following sentences: \n\n{typing_sentences}\n")
    
    input("Press Enter when you are ready to start typing...")
    start_time = time.time()
    typed_sentence = input()
    end_time = time.time()

    total_words = len(typed_sentence.split())
    total_time_minutes = (end_time - start_time) / 60
    wpm = total_words / total_time_minutes

    return wpm

def save_model(model, filename):
#   Save model to a file
    with open(filename, 'wb') as model_file:
        pickle.dump(model, model_file)

def load_model(filename):
#   Load model from a file
    with open(filename, 'rb') as model_file:
        return pickle.load(model_file)