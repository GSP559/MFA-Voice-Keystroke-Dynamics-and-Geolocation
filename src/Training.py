#Training_Final.py
import numpy as np
import random
from sklearn.svm import SVC
from Utils import extract_features_with_pitch_and_tone, record_voice, record_typing, save_model
from Google_API import get_geocode_location
api_key = 'YOUR_API_KEY'

#   Define Typing training variables
typing_training_data = []
typing_labels = []

#   Inputs for username and verified address
username = input("Please enter your unique username: ")
verified_address = input("Please enter your verified address in the following format (Street, City, State, Postal code, Country): ")
#   Get geocoded location coords for the verified address
verified_location = get_geocode_location(verified_address, api_key)

#   Record sample for the authenticated user
record_filename = f"voice_training_{username}.wav"
record_voice(record_filename, duration=15, sr=22050)
#   Load & extract features from the recorded audio
features = extract_features_with_pitch_and_tone(record_filename)

#   Typing data collection    
wpm = record_typing()
typing_training_data.append([wpm])
typing_labels.append(1)  # Label '1' for authorized typing data

#   Calculate the average and standard deviation of WPM for the authorized user
authorized_wpm = [wpm for [wpm] in typing_training_data]
average_wpm = np.mean(authorized_wpm)

# Set lowest and highest authorized WPM for dummy data training
min_authorized_wpm = min(authorized_wpm)
max_authorized_wpm = max(authorized_wpm)

# Generate random WPM for unauthorized users based on a range that doesn't overlap with the authorized user's WPM
for _ in range(5):
    if random.choice([True, False]):
        random_wpm = random.uniform(min_authorized_wpm - 20, min_authorized_wpm - 1)
    else:
        random_wpm = random.uniform(max_authorized_wpm + 1, max_authorized_wpm + 20)
   
    typing_training_data.append([random_wpm])
    typing_labels.append(0)  # Label '0' for unauthorized typing data

#   Train SVM Classifier for typing dynamics
typing_clf = SVC()
typing_clf.fit(np.array(typing_training_data), np.array(typing_labels))  # Use typing_labels here

#   Label and save the trained models with usernames
voice_model_filename = f'voice_model_{username}.pkl'
typing_model_filename = f'typing_model_{username}.pkl'
location_filename = f'location_{username}.pkl'

save_model(features, voice_model_filename)
save_model(typing_clf, typing_model_filename)
save_model(verified_location, location_filename)