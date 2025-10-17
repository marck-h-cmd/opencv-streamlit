import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class ObjectTrackingProcessor(VideoProcessorBase):
    def __init__(self):
        self.selection = None
        self.drag_start = None
        self.tracking_state = 0
        self.track_window = None
        self.hist = None
        self.scaling_factor = 0.8

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            vis = img.copy()
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)),
                             np.array((180., 255., 255.)))
            
            if self.selection:
                x0, y0, x1, y1 = self.selection
                self.track_window = (x0, y0, x1-x0, y1-y0)
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]
                
                hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                self.hist = hist.reshape(-1)
                
                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0
            
            if self.tracking_state == 1:
                self.selection = None
                
                prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                
                if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
                    try:
                        track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)
                        cv2.ellipse(vis, track_box, (0, 255, 0), 2)
                    except Exception:
                        pass
            
            return av.VideoFrame.from_ndarray(vis, format="bgr24")
        except Exception:
            return frame

def show():
    st.header("🎯 Capítulo 8: Object Tracking")
    st.markdown("---")
    
    st.subheader("⚙️ Rastreador Interactivo de Objetos")
    
    st.info("🖱️ Selecciona el área del objeto a rastrear usando los controles")
    
    col1, col2 = st.columns(2)
    with col1:
        x = st.slider("Coordenada X", 0, 640, 100)
        y = st.slider("Coordenada Y", 0, 480, 100)
    with col2:
        width = st.slider("Ancho", 50, 300, 100)
        height = st.slider("Alto", 50, 300, 100)
    
    if st.button("Iniciar Rastreo", key="start_tracking"):
        st.session_state.tracking_active = True
        st.session_state.selection_area = (x, y, x+width, y+height)
    
    if st.session_state.get('tracking_active', False):
        ctx = webrtc_streamer(
            key="object-tracking",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=ObjectTrackingProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,
        )
        
        if ctx.video_processor:
            ctx.video_processor.selection = st.session_state.get('selection_area')
            ctx.video_processor.tracking_state = 1
    
 