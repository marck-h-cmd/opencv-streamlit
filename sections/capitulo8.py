import streamlit as st
import cv2
import numpy as np

def show():
    st.header("🎯 Capítulo 8: Object Tracking")
    st.markdown("---")
    
    st.subheader("⚙️ Rastreador Interactivo de Objetos")
    
    if st.button("Iniciar Rastreador", key="start_tracker"):
        st.session_state.tracking_active = True
    
    if st.button("Detener Rastreador", key="stop_tracker"):
        st.session_state.tracking_active = False
    
    if st.session_state.get('tracking_active', False):
        st.info("🖱️ Haz clic y arrastra para seleccionar el objeto a rastrear")
        
        class ObjectTracker:
            def __init__(self):
                self.cap = cv2.VideoCapture(0)
                ret, self.frame = self.cap.read()
                self.scaling_factor = 0.8
                self.frame = cv2.resize(self.frame, None, fx=self.scaling_factor,
                                      fy=self.scaling_factor, interpolation=cv2.INTER_AREA)
                self.selection = None
                self.drag_start = None
                self.tracking_state = 0
                self.track_window = None
                self.hist = None
            
            def mouse_event(self, event, x, y, flags, param):
                x, y = np.int16([x, y])
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.drag_start = (x, y)
                    self.tracking_state = 0
                
                if self.drag_start:
                    if event == cv2.EVENT_MOUSEMOVE:
                        h, w = self.frame.shape[:2]
                        xo, yo = self.drag_start
                        x0, y0 = np.maximum(0, np.minimum([xo, yo], [x, y]))
                        x1, y1 = np.minimum([w, h], np.maximum([xo, yo], [x, y]))
                        self.selection = None
                        if x1-x0 > 0 and y1-y0 > 0:
                            self.selection = (x0, y0, x1, y1)
                    
                    elif event == cv2.EVENT_LBUTTONUP:
                        self.drag_start = None
                        if self.selection is not None:
                            self.tracking_state = 1
            
            def start_tracking(self):
                stframe = st.empty()
                stop_button = st.button("🛑 Detener Rastreo")
                
                while self.cap.isOpened() and not stop_button:
                    ret, self.frame = self.cap.read()
                    if not ret:
                        break
                    
                    self.frame = cv2.resize(self.frame, None,
                                          fx=self.scaling_factor, fy=self.scaling_factor,
                                          interpolation=cv2.INTER_AREA)
                    
                    vis = self.frame.copy()
                    hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
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
                            except:
                                pass
                    
                    frame_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    if stop_button:
                        break
                
                self.cap.release()
        
        tracker = ObjectTracker()
        tracker.start_tracking()