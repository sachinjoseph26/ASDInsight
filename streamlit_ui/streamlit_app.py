import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import requests
import cv2
from PIL import Image
from av import VideoFrame
import numpy as np

from streamlit_option_menu import option_menu

API_URL = "http://172.19.0.2:5555/predict-eyebased"
API_URL_local = "http://localhost:5555/predict-eyebased"
st.set_page_config(layout="wide")
# Function to send image to the API
def predict_image(file):
 # Update the URL if necessary
    files = {"file": file}
    response = requests.post(API_URL, files=files)
    if response.status_code == 200:
        return response.json().get("prediction")
    else:
        return f"Error: {response.json().get('error')}"

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
        # Section for capturing image from the camera
        #st.set_page_config(layout="wide")
        st.subheader('Capture from Camera')

        # Constants
        ORN = 7

        # Initialize state variables
        if 'k' not in st.session_state:
            st.session_state.k = 1
            st.session_state.sumx = 0
            st.session_state.sumy = 0
            st.session_state.eye_x_positions = []
            st.session_state.eye_y_positions = []
            st.session_state.running = True

        # Function to process the frame and detect eyes
        def process_frame(image):
            k = st.session_state.k
            sumx = st.session_state.sumx
            sumy = st.session_state.sumy

            image = cv2.flip(image, 1)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(eyes) > 0:
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    roi = gray[ey:ey + eh, ex:ex + ew]
                    roi_color = image[ey:ey + eh, ex:ex + ew]

                    eye_blur = cv2.bilateralFilter(roi, 10, 195, 195)
                    img_blur = cv2.Canny(eye_blur, 10, 30)

                    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 20, param1=200, param2=10, minRadius=0, maxRadius=0)

                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        for i in circles[0, :]:
                            cv2.circle(roi_color, (i[0], i[1]), i[2], (0, 255, 0), 2)
                            cv2.circle(roi_color, (i[0], i[1]), 2, (0, 0, 255), 3)

                            if k == ORN:
                                k = 1
                                sumx /= ORN
                                sumy /= ORN
                                eye_x_p = round((sumx - 145), 2)
                                eye_y_p = round((sumy - 145), 2)
                                st.session_state.eye_x_positions.append(eye_x_p)
                                st.session_state.eye_y_positions.append(eye_y_p)
                                sumx = 0
                                sumy = 0
                            else:
                                sumx += i[0]
                                sumy += i[1]
                                k += 1

                            st.session_state.k = k
                            st.session_state.sumx = sumx
                            st.session_state.sumy = sumy

            return image

        # Streamlit layout
        video_placeholder = st.empty()
        coord_placeholder = st.empty()

        # Stop button
        if st.button("Stop"):
            st.session_state.running = False

        # Main loop
        cap = cv2.VideoCapture(0)

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break

            # Process the frame
            processed_frame = process_frame(frame)

            # Convert the frame to RGB (streamlit expects RGB)
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Update the video feed
            video_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)

            # Update the eye coordinates
            coord_placeholder.write("Latest Eye Coordinates:")
            if st.session_state.eye_x_positions and st.session_state.eye_y_positions:
                latest_x, latest_y = st.session_state.eye_x_positions[-1], st.session_state.eye_y_positions[-1]
                coord_placeholder.write(f"X: {latest_x}, Y: {latest_y}")

            # Break loop if stop button is pressed
            if not st.session_state.running:
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Display final results
        st.write("Eye Tracking Stopped")
        st.write("Eye Coordinate History:")
        for i, (x, y) in enumerate(zip(st.session_state.eye_x_positions, st.session_state.eye_y_positions)):
            st.write(f"Frame {i + 1}: X: {x}, Y: {y}")
elif selected == 'Q-CHAT Based':
    st.header('Q-CHAT Based Prediction')
    st.write("Q-CHAT prediction functionality is under construction.")
    
elif selected == 'Video Capture':
    st.header('Video Capture Prediction')
    st.write("Video capture prediction functionality is under construction.")