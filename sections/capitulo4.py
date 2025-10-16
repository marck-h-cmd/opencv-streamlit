import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class BodyPartsDetector(VideoProcessorBase):
    def __init__(self):
        self.detection_mode = "Detección de rostros"
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_size = 30
        self.blur_value = 5
        self.param1 = 50
        self.param2 = 30
        self.min_radius = 10
        self.max_radius = 50
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.detection_mode == "Detección de rostros":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=self.scale_factor, 
                                                     minNeighbors=self.min_neighbors, 
                                                     minSize=(self.min_size, self.min_size))
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, 'Rostro', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        elif self.detection_mode == "Detección de ojos":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=self.scale_factor, 
                                                       minNeighbors=self.min_neighbors, 
                                                       minSize=(self.min_size, self.min_size))
                for (x_eye, y_eye, w_eye, h_eye) in eyes:
                    center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
                    radius = int(0.3 * (w_eye + h_eye))
                    cv2.circle(roi_color, center, radius, (0, 255, 0), 2)
                    cv2.putText(roi_color, 'Ojo', (x_eye, y_eye-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        elif self.detection_mode == "Detección de boca":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                mouths = self.mouth_cascade.detectMultiScale(roi_gray, scaleFactor=self.scale_factor, 
                                                           minNeighbors=self.min_neighbors, 
                                                           minSize=(self.min_size, self.min_size))
                for (x_mouth, y_mouth, w_mouth, h_mouth) in mouths:
                    cv2.rectangle(roi_color, (x_mouth, y_mouth), (x_mouth+w_mouth, y_mouth+h_mouth), (0, 0, 255), 2)
                    cv2.putText(roi_color, 'Boca', (x_mouth, y_mouth-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        elif self.detection_mode == "Detección de pupilas":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in eyes:
                roi_gray = gray[y:y+h, x:x+w]
                blurred = cv2.medianBlur(roi_gray, self.blur_value)
                circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                                         param1=self.param1, param2=self.param2,
                                         minRadius=self.min_radius, maxRadius=self.max_radius)
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    for (cx, cy, r) in circles:
                        cv2.circle(img, (x+cx, y+cy), r, (0, 255, 255), 2)
                        cv2.circle(img, (x+cx, y+cy), 2, (0, 255, 255), 3)
                        cv2.putText(img, 'Pupila', (x+cx-20, y+cy-r-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def show():
    st.header("👤 Capítulo 4: Body Parts Detection")
    st.markdown("---")
    
    st.subheader("⚙️ Selecciona la parte del cuerpo a detectar")
    
    col_buttons = st.columns(4)
    with col_buttons[0]:
        if st.button("Rostros", use_container_width=True):
            st.session_state.detection_mode = "Detección de rostros"
    with col_buttons[1]:
        if st.button("Ojos", use_container_width=True):
            st.session_state.detection_mode = "Detección de ojos"
    with col_buttons[2]:
        if st.button("Boca", use_container_width=True):
            st.session_state.detection_mode = "Detección de boca"
    with col_buttons[3]:
        if st.button("Pupilas", use_container_width=True):
            st.session_state.detection_mode = "Detección de pupilas"
    
    if 'detection_mode' not in st.session_state:
        st.session_state.detection_mode = "Detección de rostros"
    
    st.markdown("---")
    
    st.info(f"🔴 **{st.session_state.detection_mode} en Tiempo Real**")
    
    if st.session_state.detection_mode == "Detección de rostros":
        scale_factor = st.slider("Scale Factor", 1.01, 1.5, 1.1, 0.01)
        min_neighbors = st.slider("Min Neighbors", 1, 10, 5)
        min_size = st.slider("Min Size", 20, 100, 30)
        
        ctx = webrtc_streamer(
            key="face-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=BodyPartsDetector,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if ctx.video_processor:
            ctx.video_processor.detection_mode = st.session_state.detection_mode
            ctx.video_processor.scale_factor = scale_factor
            ctx.video_processor.min_neighbors = min_neighbors
            ctx.video_processor.min_size = min_size
    
    elif st.session_state.detection_mode == "Detección de ojos":
        scale_factor = st.slider("Scale Factor", 1.01, 1.5, 1.1, 0.01, key="eyes_scale")
        min_neighbors = st.slider("Min Neighbors", 1, 10, 5, key="eyes_neighbors")
        min_size = st.slider("Min Size", 10, 50, 20, key="eyes_size")
        
        ctx = webrtc_streamer(
            key="eye-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=BodyPartsDetector,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if ctx.video_processor:
            ctx.video_processor.detection_mode = st.session_state.detection_mode
            ctx.video_processor.scale_factor = scale_factor
            ctx.video_processor.min_neighbors = min_neighbors
            ctx.video_processor.min_size = min_size
    
    elif st.session_state.detection_mode == "Detección de boca":
        scale_factor = st.slider("Scale Factor", 1.01, 1.5, 1.1, 0.01, key="mouth_scale")
        min_neighbors = st.slider("Min Neighbors", 1, 10, 5, key="mouth_neighbors")
        min_size = st.slider("Min Size", 20, 80, 30, key="mouth_size")
        
        ctx = webrtc_streamer(
            key="mouth-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=BodyPartsDetector,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if ctx.video_processor:
            ctx.video_processor.detection_mode = st.session_state.detection_mode
            ctx.video_processor.scale_factor = scale_factor
            ctx.video_processor.min_neighbors = min_neighbors
            ctx.video_processor.min_size = min_size
    
    elif st.session_state.detection_mode == "Detección de pupilas":
        blur_value = st.slider("Desenfoque", 1, 15, 5, key="pupil_blur")
        param1 = st.slider("Param1", 10, 100, 50, key="pupil_param1")
        param2 = st.slider("Param2", 10, 100, 30, key="pupil_param2")
        min_radius = st.slider("Radio Min", 1, 50, 10, key="pupil_min")
        max_radius = st.slider("Radio Max", 10, 100, 50, key="pupil_max")
        
        ctx = webrtc_streamer(
            key="pupil-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=BodyPartsDetector,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if ctx.video_processor:
            ctx.video_processor.detection_mode = st.session_state.detection_mode
            ctx.video_processor.blur_value = blur_value
            ctx.video_processor.param1 = param1
            ctx.video_processor.param2 = param2
            ctx.video_processor.min_radius = min_radius
            ctx.video_processor.max_radius = max_radius