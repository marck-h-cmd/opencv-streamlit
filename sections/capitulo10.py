import streamlit as st
import cv2
import numpy as np

def show():
    st.header("🌐 Capítulo 10: Augmented Reality")
    st.markdown("---")
    
    st.subheader("⚙️ Seguimiento de Objetos Planos")
    
    if st.button("Iniciar Seguimiento AR", key="start_ar"):
        st.session_state.ar_tracking_active = True
    
    if st.button("Detener Seguimiento", key="stop_ar"):
        st.session_state.ar_tracking_active = False
    
    if st.session_state.get('ar_tracking_active', False):
        st.info("🖱️ Haz clic y arrastra para seleccionar el objeto plano a seguir")
        
        class ROISelector:
            def __init__(self, win_name, init_frame, callback_func):
                self.callback_func = callback_func
                self.selected_rect = None
                self.drag_start = None
                self.tracking_state = 0
                self.event_params = {"frame": init_frame}
            
            def mouse_event(self, event, x, y, flags, param):
                x, y = np.int16([x, y])
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.drag_start = (x, y)
                    self.tracking_state = 0
                
                if self.drag_start:
                    if event == cv2.EVENT_MOUSEMOVE:
                        h, w = param["frame"].shape[:2]
                        xo, yo = self.drag_start
                        x0, y0 = np.maximum(0, np.minimum([xo, yo], [x, y]))
                        x1, y1 = np.minimum([w, h], np.maximum([xo, yo], [x, y]))
                        self.selected_rect = None
                        if x1-x0 > 0 and y1-y0 > 0:
                            self.selected_rect = (x0, y0, x1, y1)
                    
                    elif event == cv2.EVENT_LBUTTONUP:
                        self.drag_start = None
                        if self.selected_rect is not None:
                            self.callback_func(self.selected_rect)
                            self.selected_rect = None
                            self.tracking_state = 1
            
            def draw_rect(self, img, rect):
                if not rect: 
                    return False
                x_start, y_start, x_end, y_end = rect
                cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                return True
        
        cap = cv2.VideoCapture(0)
        ret, init_frame = cap.read()
        
        roi_selector = ROISelector("AR Tracking", init_frame, lambda rect: None)
        st.session_state.selected_roi = None
        
        def roi_callback(rect):
            st.session_state.selected_roi = rect
        
        roi_selector.callback_func = roi_callback
        
        stframe = st.empty()
        stop_button = st.button("🛑 Detener")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            if st.session_state.get('selected_roi'):
                x0, y0, x1, y1 = st.session_state.selected_roi
                
                roi = frame[y0:y1, x0:x1]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                corners = cv2.goodFeaturesToTrack(gray_roi, 100, 0.01, 10)
                
                if corners is not None:
                    corners = np.int0(corners)
                    for corner in corners:
                        x, y = corner.ravel()
                        cv2.circle(frame, (x0+x, y0+y), 3, (0, 0, 255), -1)
                
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                
                center_x = (x0 + x1) // 2
                center_y = (y0 + y1) // 2
                
                cv2.putText(frame, "OBJETO SEGUIDO", (center_x-80, y0-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.line(frame, (center_x-20, center_y), (center_x+20, center_y), (255, 0, 0), 2)
                cv2.line(frame, (center_x, center_y-20), (center_x, center_y+20), (255, 0, 0), 2)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)
            
            if stop_button:
                break
        
        cap.release()