import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import requests
import cv2
from PIL import Image
from av import VideoFrame
import json
import numpy as np
import os
from av import VideoFrame
from streamlit_option_menu import option_menu
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

st.set_page_config(layout="wide")
API_URL_old = "http://asd-insight:7801/predict-eyebased"
API_URL = "http://localhost:7811/predict-eyebased"
QCHAT_API_URL = "http://localhost:7811/qchat-screening/predict-qchat-asdrisk"

# Function to send image to the API
def predict_image(file):
 # Update the URL if necessary
    files = {"file": file}
    response = requests.post(API_URL, files=files)
    if response.status_code == 200:
        return response.json().get("prediction")
    else:
        return f"Error: {response.json().get('error')}"
def predict_asd_risk(api_url, file_path):
    try:
        # Open the file in binary mode for uploading
        with open(file_path, 'rb') as file:
            # Create a dictionary for the file to be sent as multipart/form-data
            files = {'file': (file_path, file, 'application/json')}
            
            # Make the POST request
            response = requests.post(api_url, files=files)
        
        # Check the response status
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code} - {response.text}"
    
    except FileNotFoundError:
        return "Error: The specified file was not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"
# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        "ASD Insight",
        ["Eye Tracking", "Q Chat Based", "Video Capture"],
        icons=["eye", "chat", "camera-video"],
        menu_icon="cast",
        default_index=0
    )

st.title("ASD Insight - Autism Spectrum Disorder Detection")

if selected == 'Eye Tracking':
    st.header('Eye Tracking Prediction')

    # Create two options: Upload an image file or Capture from camera
    option = st.radio("Choose input method:", ["Upload Image File", "Capture from Camera"])

    if option == "Upload Image File":
        # Section for uploading an image file
        st.subheader('Upload an Image File')
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            prediction = predict_image(uploaded_file)
            st.write(f'Prediction: {prediction}')

    elif option == "Capture from Camera":
        st.subheader('Capture from Camera')

        class EyeTrackingProcessor(VideoProcessorBase):
            def __init__(self):
                self.face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
                self.eye_coordinates = []  # To store eye coordinates

            def recv(self, frame: VideoFrame) -> VideoFrame:
                image = frame.to_ndarray(format="bgr24")
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_image)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Extract coordinates for the left and right eyes
                        left_eye_coords = [(int(face_landmarks.landmark[i].x * image.shape[1]), 
                                            int(face_landmarks.landmark[i].y * image.shape[0])) 
                                           for i in [33, 133]]
                        right_eye_coords = [(int(face_landmarks.landmark[i].x * image.shape[1]), 
                                             int(face_landmarks.landmark[i].y * image.shape[0])) 
                                            for i in [362, 263]]
                        
                        # Draw circles on detected eye landmarks
                        for (x, y) in left_eye_coords + right_eye_coords:
                            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                        # Calculate and store the average position of the left and right eyes
                        left_eye_center = np.mean(left_eye_coords, axis=0).astype(int)
                        right_eye_center = np.mean(right_eye_coords, axis=0).astype(int)
                        cv2.circle(image, tuple(left_eye_center), 3, (0, 0, 255), -1)
                        cv2.circle(image, tuple(right_eye_center), 3, (0, 0, 255), -1)

                        # Append to eye_coordinates
                        self.eye_coordinates.append((tuple(left_eye_center), tuple(right_eye_center)))
                
                return VideoFrame.from_ndarray(image, format="bgr24")

        webrtc_ctx = webrtc_streamer(key="eye-tracking", video_processor_factory=EyeTrackingProcessor)

        if webrtc_ctx.video_processor:
            st.write("Eye Tracking is running...")

            if st.button("Show Captured Coordinates"):
                eye_coords = webrtc_ctx.video_processor.eye_coordinates
                if eye_coords:
                    st.write("Captured Eye Coordinates History:")
                    for i, (left, right) in enumerate(eye_coords):
                        st.write(f"Frame {i + 1}: Left Eye: {left}, Right Eye: {right}")
                else:
                    st.write("No coordinates captured yet.")

elif selected == 'Q Chat Based':
    st.header('Q-CHAT Assessment Form')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age',min_value=0, format="%d")
        birth_weight = st.number_input('Birth weight',min_value=0.0, format="%.2f")
        mother_education = st.selectbox('Mother educational qualification', ['Primary/Vocational', 'High School', 'Higher Studies'],index=None)
        preterm = st.radio("Does the child was born pre-term", ["Yes", "No"], index=None)
    with col2:
        sex = st.radio("Sex", ["Male", "Female"], index=None)
        has_siblings = st.radio("Does the child has siblings", ["Yes", "No"], index=None)
        if has_siblings == 'Yes':
            sibling_with_asd = st.radio("Does the child\'s sibling has ASD", ["Yes", "No"], index=None)
        else:
            sibling_with_asd = None

    with col3:
        if has_siblings == 'Yes':
            num_siblings = st.number_input('Number of siblings',min_value=1, format="%d")
        else:
            num_siblings = None

    # Additional questions with specified scales
    st.subheader('Additional Behavioral Questions')

    additional_questions = {
        'Does the child looks when called by name': [('always',0), ('usually',1), ('sometimes',2), ('rarely',3), ('never',4)],
        'Does the child maintains eye to eye contact': [('very easy',0), ('fairly easy',1), ('fairly difficult',2), ('very difficult',3), ('impossible',4)],
        'Does the child can line-up objects': [('never',0), ('seldom',1), ('sometimes',2), ('usually',3), ('always',4)],
        'Does others can understand the child’s speech':[('always',0), ('usually',1), ('sometimes',2), ('seldom',3), ('never or never says',4)],
        'Does the child points to indicate objects': [('many times/day',0), ('several times/day',1), ('several times/week',2), ('less than once/week',3), ('never',4)],

        'Does the child shares interest by pointing to scenes': [('many times/day',0), ('several times/day',1), ('several times/week',2), ('less than once/week',3), ('never',4)],
        'Does the child focuses on spinning objects': [('less than a minute',0),  ('a few minutes',1), ('10 minutes',2), ('half an hour',3),  ('a few hours',4)],
        'Count of words that the child uses': [('over 100 words',0), ('51-100 words',1), ('10-50 words',2), ('less than 10 words',3), ('None - hasn\'t started talking yet',4)],
        'Does the child can play pretend': [('many times/day',0), ('several times/day',1), ('several times/week',2), ('less than once/week',3), ('never',4)],
        'Does the child can track what the parents see': [('many times/day',0), ('several times/day',1), ('several times/week',2), ('less than once/week',3), ('never',4)],

        'Does the child licks objects': [('never',0), ('less than once a week',1), ('several times/week',2), ('several times/day',3), ('many times/day',4)],
        'Does the child uses their hands as tools': [('never',0), ('less than once a week',1), ('several times/week',2), ('several times/day',3), ('many times/day',4)],
        'Does the child tiptoes': [('never',0), ('seldom',1), ('sometimes',2), ('usually',3), ('always',4)],
        'Does the child can accept changes in routines': [('very easy',0), ('fairly easy',1), ('fairly difficult',2), ('very difficult',3), ('impossible',4)],
        'Does the child tries to comfort others': [('always',0), ('usually',1), ('sometimes',2), ('rarely',3), ('never',4)],

        'Does the child does the same thing repeatedly': [('never',0), ('less than once a week',1), ('several times/week',2), ('several times/day',3), ('many times/day',4)],
        'The child’s first words': [('very typical',0), ('pretty typical',1), ('a bit unusual',2), ('very unusual',3), ('doesn\'t say',4)], 
        'Does the child mimics/echos': [('never',0), ('less than once a week',1), ('several times/week',2), ('several times/day',3), ('many times/day',4)],
        'Does the child makes simple gestures': [('many times/day',0), ('several times/day',1), ('several times/week',2), ('less than once/week',3), ('never',4)],
        'Does the child makes unusual finger movements near the eyes': [('never',0), ('less than once a week',1), ('several times/week',2), ('several times/day',3), ('many times/day',4)],

        'Dpes the child checks the parent’s reaction': [('always',0), ('usually',1), ('sometimes',2), ('rarely',3), ('never',4)],
        'Does the child maintains interest': [('few minutes',0), ('10 minutes',1) , ('half an hour',2), ('a few hours',3),  ('most of the day',4)],
        'Does the child twiddles with objects': [('never',0), ('less than once a week',1), ('several times/week',2), ('several times/day',3), ('many times/day',4)],
        'Does the child sensitivity to noise': [('never',0), ('seldom',1), ('sometimes',2), ('usually',3), ('always',4)],
        'Does the child stares at something without any intent': [('never',0), ('less than once a week',1), ('several times/week',2), ('several times/day',3), ('many times/day',4)]
    }




    selected_values = {}
    sum_qchat = 0
    for index, (question, options) in enumerate(additional_questions.items()):
        num = index+1
        key = 'qchat'+str(num)+'recode'
        selected_option = st.selectbox(
            question,
            options = [opt[0] for opt in options],
            key=key,
            index=None
        )
        # Store the selected value in the dictionary
        value = next(
            (value for label, value in options if label == selected_option),
            None
        )
        selected_values[key] = value
    if st.button('Predict ASD Risk'):
        additional_responses = selected_values
        sum_qchat = 0
        answered = 'No'
        for question_index,value in selected_values.items():
            if value is not None:
                answered = 'Yes'
                sum_qchat += value
            else:
                answered = 'No'
                break
                
        if (age is None or
             birth_weight is None or
             mother_education is None or
             preterm is None or
             sex is None or 
             has_siblings is None or
            (has_siblings == 'Yes' and (num_siblings is None or sibling_with_asd is None)) or
            answered == 'No'):
            st.error("Please enter all mandatory fields and provide answers to the behavioral questions")
        else:
            data = {
                "age": age,
                "sex": 1 if sex == 'Male' else 0,
                "preterm": 1 if preterm == 'Yes' else 0,
                "birthweight": birth_weight,
                "siblings_yesno": 1 if has_siblings == 'Yes' else 0,
                "siblings_number": 0 if has_siblings == 'No' else num_siblings,
                "mothers_education": 1 if mother_education == 'Primary/Vocational' else 
                        2 if mother_education == 'High School' else 
                        3 if mother_education == 'Higher Studies' else None,
                "sibling_withASD": 1 if has_siblings == 'Yes' and sibling_with_asd == 'Yes' else 0,
                "Sum_QCHAT": sum_qchat,
                **additional_responses
            }
            #st.write(data)
            file_path = 'qchat_assessment.json'
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            response = predict_asd_risk(QCHAT_API_URL,file_path)
             # Display the result in bold
            result = response.get("prediction")
            st.markdown(f"<h2 style='font-size:24px;'>Analysis Results: {result}</h2>", unsafe_allow_html=True)
            # Optionally, remove the file after upload
            if os.path.exists(file_path):
                os.remove(file_path)
elif selected == 'Video Capture':
    st.header('Video Capture Prediction')
    st.write("Video capture prediction functionality is under construction.")
