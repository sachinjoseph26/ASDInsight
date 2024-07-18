import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import requests
import cv2
from PIL import Image
from av import VideoFrame

from streamlit_option_menu import option_menu

API_URL = "http://asd-insight-api:5555/predict-eyebased"
API_URL_local = "http://localhost:5555/predict-eyebased"

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
        st.subheader('Capture Image from Camera')
        
        class VideoProcessor(VideoTransformerBase):
            def __init__(self):
                self._last_frame = None

            def recv(self, frame):
                self._last_frame = frame.to_ndarray(format="bgr24")
                print("Frame captured")
                return frame
            
            def get_last_frame(self):
                return self._last_frame

        webrtc_ctx = webrtc_streamer(key="key", video_processor_factory=VideoProcessor, 
                                     media_stream_constraints={"video": True, "audio": False})
        
        if st.button("Capture Image"):
            if webrtc_ctx.video_processor:
                img = webrtc_ctx.video_processor.get_last_frame()
                if img is not None:
                    print("Image captured from webcam")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img)
                    img_pil.save("captured_image.png")  # Save the captured image
                
                    with open("captured_image.png", "rb") as img_file:
                        prediction = predict_image(img_file)
                        st.image(img, caption='Captured Image', use_column_width=True)
                        st.write(f'Prediction: {prediction}')       
                else:
                    st.write("No frame captured from the webcam.")
                    print("No frame captured from the webcam.")

elif selected == 'Q-CHAT Based':
    st.header('Q-CHAT Based Prediction')
    st.write("Q-CHAT prediction functionality is under construction.")
    
elif selected == 'Video Capture':
    st.header('Video Capture Prediction')
    st.write("Video capture prediction functionality is under construction.")