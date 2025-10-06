import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

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
        
        if st.button("Iniciar Detección de Rostros", key="start_faces"):
            st.session_state.face_detection_active = True
        
        if st.button("Detener Detección", key="stop_faces"):
            st.session_state.face_detection_active = False
        
        if st.session_state.get('face_detection_active', False):
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            if face_cascade.empty():
                st.error("No se pudo cargar el clasificador de rostros")
            else:
                cap = cv2.VideoCapture(0)
                stframe = st.empty()
                stop_button = st.button("🛑 Detener")
                
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(min_size, min_size))
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, 'Rostro', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    if stop_button:
                        break
                
                cap.release()
    
    elif st.session_state.detection_mode == "Detección de ojos":
        scale_factor = st.slider("Scale Factor", 1.01, 1.5, 1.1, 0.01, key="eyes_scale")
        min_neighbors = st.slider("Min Neighbors", 1, 10, 5, key="eyes_neighbors")
        min_size = st.slider("Min Size", 10, 50, 20, key="eyes_size")
        
        if st.button("Iniciar Detección de Ojos", key="start_eyes"):
            st.session_state.eye_detection_active = True
        
        if st.button("Detener Detección", key="stop_eyes"):
            st.session_state.eye_detection_active = False
        
        if st.session_state.get('eye_detection_active', False):
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            if face_cascade.empty() or eye_cascade.empty():
                st.error("No se pudieron cargar los clasificadores")
            else:
                cap = cv2.VideoCapture(0)
                stframe = st.empty()
                stop_button = st.button("🛑 Detener")
                
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1)
                    
                    for (x, y, w, h) in faces:
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = frame[y:y+h, x:x+w]
                        
                        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(min_size, min_size))
                        
                        for (x_eye, y_eye, w_eye, h_eye) in eyes:
                            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
                            radius = int(0.3 * (w_eye + h_eye))
                            cv2.circle(roi_color, center, radius, (0, 255, 0), 2)
                            cv2.putText(roi_color, 'Ojo', (x_eye, y_eye-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    if stop_button:
                        break
                
                cap.release()
    
    elif st.session_state.detection_mode == "Detección de boca":
        scale_factor = st.slider("Scale Factor", 1.01, 1.5, 1.1, 0.01, key="mouth_scale")
        min_neighbors = st.slider("Min Neighbors", 1, 10, 5, key="mouth_neighbors")
        min_size = st.slider("Min Size", 20, 80, 30, key="mouth_size")
        
        if st.button("Iniciar Detección de Boca", key="start_mouth"):
            st.session_state.mouth_detection_active = True
        
        if st.button("Detener Detección", key="stop_mouth"):
            st.session_state.mouth_detection_active = False
        
        if st.session_state.get('mouth_detection_active', False):
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            
            if face_cascade.empty() or mouth_cascade.empty():
                st.error("No se pudieron cargar los clasificadores")
            else:
                cap = cv2.VideoCapture(0)
                stframe = st.empty()
                stop_button = st.button("🛑 Detener")
                
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1)
                    
                    for (x, y, w, h) in faces:
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = frame[y:y+h, x:x+w]
                        
                        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(min_size, min_size))
                        
                        for (x_mouth, y_mouth, w_mouth, h_mouth) in mouths:
                            cv2.rectangle(roi_color, (x_mouth, y_mouth), (x_mouth+w_mouth, y_mouth+h_mouth), (0, 0, 255), 2)
                            cv2.putText(roi_color, 'Boca', (x_mouth, y_mouth-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    if stop_button:
                        break
                
                cap.release()
    
    elif st.session_state.detection_mode == "Detección de pupilas":
        blur_value = st.slider("Desenfoque", 1, 15, 5, key="pupil_blur")
        param1 = st.slider("Param1", 10, 100, 50, key="pupil_param1")
        param2 = st.slider("Param2", 10, 100, 30, key="pupil_param2")
        min_radius = st.slider("Radio Min", 1, 50, 10, key="pupil_min")
        max_radius = st.slider("Radio Max", 10, 100, 50, key="pupil_max")
        
        if st.button("Iniciar Detección de Pupilas", key="start_pupils"):
            st.session_state.pupil_detection_active = True
        
        if st.button("Detener Detección", key="stop_pupils"):
            st.session_state.pupil_detection_active = False
        
        if st.session_state.get('pupil_detection_active', False):
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            if eye_cascade.empty():
                st.error("No se pudo cargar el clasificador de ojos")
            else:
                cap = cv2.VideoCapture(0)
                stframe = st.empty()
                stop_button = st.button("🛑 Detener")
                
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
                    
                    for (x, y, w, h) in eyes:
                        roi_gray = gray[y:y+h, x:x+w]
                        
                        blurred = cv2.medianBlur(roi_gray, blur_value)
                        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                                                 param1=param1, param2=param2,
                                                 minRadius=min_radius, maxRadius=max_radius)
                        
                        if circles is not None:
                            circles = np.round(circles[0, :]).astype("int")
                            for (cx, cy, r) in circles:
                                cv2.circle(frame, (x+cx, y+cy), r, (0, 255, 255), 2)
                                cv2.circle(frame, (x+cx, y+cy), 2, (0, 255, 255), 3)
                                cv2.putText(frame, 'Pupila', (x+cx-20, y+cy-r-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    if stop_button:
                        break
                
                cap.release()
    
    st.markdown("---")
    
    code_demo = """
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
if eye_cascade.empty():
    raise IOError('Unable to load the eye cascade classifier xml file')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1)
    
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (x_eye,y_eye,w_eye,h_eye) in eyes:
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            radius = int(0.3 * (w_eye + h_eye))
            cv2.circle(roi_color, center, radius, (0, 255, 0), 2)
    
    cv2.imshow('Eye Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
    """
    
    with st.expander("💻 Código para detección en tiempo real"):
        st.code(code_demo, language="python")
    
    st.markdown("---")
    st.subheader("📖 Detecciones disponibles:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Detección de rostros**: Usando Haar Cascades
        - **Detección de ojos**: Dentro de las regiones faciales
        - **Haar Cascades**: Algoritmo de detección en tiempo real
        """)
    
    with col2:
        st.markdown("""
        - **Detección de boca**: Reconocimiento de sonrisas
        - **Detección de pupilas**: Usando Hough Circles
        - **Imágenes integrales**: Optimización para detección rápida
        """)