import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class ARTrackingProcessor(VideoProcessorBase):
    def __init__(self):
        self.selected_roi = None
        self.tracking_active = False

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            if self.selected_roi and self.tracking_active:
                x0, y0, x1, y1 = self.selected_roi
                
                roi = img[y0:y1, x0:x1]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                corners = cv2.goodFeaturesToTrack(gray_roi, 100, 0.01, 10)
                
                if corners is not None:
                    corners = np.int0(corners)
                    for corner in corners:
                        x, y = corner.ravel()
                        cv2.circle(img, (x0+x, y0+y), 3, (0, 0, 255), -1)
                
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                
                center_x = (x0 + x1) // 2
                center_y = (y0 + y1) // 2
                
                cv2.putText(img, "OBJETO SEGUIDO", (center_x-80, y0-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.line(img, (center_x-20, center_y), (center_x+20, center_y), (255, 0, 0), 2)
                cv2.line(img, (center_x, center_y-20), (center_x, center_y+20), (255, 0, 0), 2)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception:
            return frame

def show():
    st.header("🌐 Capítulo 10: Augmented Reality")
    st.markdown("---")
    
    st.subheader("⚙️ Seguimiento de Objetos Planos")
    
    st.info("🎯 Selecciona el área del objeto plano a seguir")
    
    col1, col2 = st.columns(2)
    with col1:
        x = st.slider("Coordenada X", 0, 640, 100)
        y = st.slider("Coordenada Y", 0, 480, 100)
    with col2:
        width = st.slider("Ancho", 50, 300, 150)
        height = st.slider("Alto", 50, 300, 150)
    
    if st.button("Iniciar Seguimiento AR", key="start_ar"):
        st.session_state.ar_tracking_active = True
        st.session_state.selected_roi = (x, y, x+width, y+height)
    
    if st.session_state.get('ar_tracking_active', False):
        ctx = webrtc_streamer(
            key="ar-tracking",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=ARTrackingProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,
        )
        
        if ctx.video_processor:
            ctx.video_processor.selected_roi = st.session_state.get('selected_roi')
            ctx.video_processor.tracking_active = True
    
   