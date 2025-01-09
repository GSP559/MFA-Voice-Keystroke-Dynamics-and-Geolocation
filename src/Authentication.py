from Utils import Comparison, extract_features_with_pitch_and_tone, record_voice, record_typing, load_model, save_model
from Google_API import get_current_location, calculate_distance
api_key = 'YOUR_API_KEY'  
username_to_test = input("Please enter your username: ")

def authenticate_user(username, api_key):
#   Define model names     
    voice_model_filename = f'voice_model_{username}.pkl'
    typing_model_filename = f'typing_model_{username}.pkl'
    location_filename = f'location_{username}.pkl'
    voice_auth_model_filename = f'voice_auth_model_{username}.pkl'
    voice_auth_filename = f'voice_auth_{username}.wav'

#   Load the trained models and verified location
    typing_clf = load_model(typing_model_filename)
    verified_location = load_model(location_filename)
    voice_model = load_model(voice_model_filename)

#   Verify current location
    current_location = get_current_location(api_key)
    distance = calculate_distance(verified_location['longitude'], verified_location['latitude'],
                                  current_location['longitude'], current_location['latitude'])

#   Location verification threshold (1000 meters)
    location_threshold = 1000000

#   Validate location or else exit
    if distance > location_threshold:
        print("Location verification failed.")
        return False

#   Proceed with voice and typing authentication
#   Record the user's voice
    record_voice(voice_auth_filename, duration=15, sr=22050)  
#   Extract features and save & load auth model
    auth_features = extract_features_with_pitch_and_tone (voice_auth_filename)
    save_model(auth_features, voice_auth_model_filename)
    auth_model = load_model(voice_auth_model_filename)
    
    #HERE IS WHERE THE VOICE COMPARISON IS MADE IF WE WANT TO USE THE DECISION FUNCTION THEN WE NEED TO PULL SOME VALUE FROM CONFIDENCE FUNCTION AND USE THAT IN DECISION FUNCTION
    voice_confidence = Comparison (voice_model, auth_model) #   Uses the value of avg from the decision function as the confidence value
#   Record the user's typing
    typing_speed = record_typing()
    
#   Predict with the typing classifier
    typing_confidence = typing_clf.decision_function([[typing_speed]])[0] 
#   Set weights for each classifier (can adjust these)
    voice_weight = 0.6
    typing_weight = 0.4

#   Calculate the weighted decision score
    decision_score = voice_weight * voice_confidence + typing_weight * typing_confidence
#    Decision threshold
    decision_threshold = 0.7  # Adjust based on your system's performance and desired security level

    print(voice_confidence, typing_confidence, decision_score) # CAN REMOVE. USED FOR FINE TUNING AND TESTING

#   Make a decision based on the decision score
    if decision_score > decision_threshold:
        return True  # User is authenticated
    else:
        return False  # User is not authenticated

#   Authentication call
is_authenticated = authenticate_user(username_to_test, api_key)
print("User Authenticated" if is_authenticated else "User NOT Authenticated")
