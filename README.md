Multifactor Authentication System: README

Features
Voice Recognition: Utilizes Mel-Frequency Cepstral Coefficients (MFCC), pitch, and tone analysis for precise user identification.
Keystroke Dynamics: Measures typing speed and patterns using a Support Vector Machine (SVM) classifier to distinguish between users.
Geolocation Verification: Employs the Google Geolocation API to confirm the user’s location within a predefined radius of their verified address.
Weighted Decision System: Combines outputs from voice, typing, and location checks into a weighted decision score for authentication.
Key Files
1. Authentication.py
Handles the user authentication process by:

Loading pre-trained models for voice, typing, and location verification.
Recording and analyzing user inputs.
Computing a weighted decision score to authenticate users.
2. Training.py
Used to train models for:

Voice Authentication: Records and extracts unique voice features.
Typing Patterns: Collects typing speed data and trains an SVM model.
Geolocation: Geocodes the verified user address for location verification.
3. Utilities.py
Includes utility functions for:

Feature extraction (MFCC, pitch, and tone).
Voice and typing data recording.
Model saving and loading.
Geolocation calculations and comparisons.
Dependencies
Python libraries:
numpy
scikit-learn
librosa
sounddevice
soundfile
pickle
Google Geolocation API Key (replace YOUR_API_KEY in the code).
How to Run
Setup:

Install the required Python libraries.
Replace YOUR_API_KEY with your Google API Key in the code files.
Training:

Run Training.py.
Enter the required inputs (username, verified address, voice sample).
Models for voice, typing, and geolocation will be saved locally.
Authentication:

Run Authentication.py.
Enter the username for verification.
Provide real-time voice and typing inputs.
The system calculates a decision score to determine authentication success.
