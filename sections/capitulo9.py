import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def show():
    st.header("🤖 Capítulo 9: Object Recognition")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Sube una imagen", type=['png', 'jpg', 'jpeg'], key="ch9")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        st.session_state.image_loaded = True
    else:
        st.session_state.image_loaded = False
    
    st.subheader("⚙️ Reconocimiento de Objetos")
    
    if st.session_state.get('image_loaded', False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📷 Imagen Original**")
            st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with col2:
            st.write("**✨ Detección de Características**")
            
            detector_type = st.selectbox("Selecciona el detector:", ["Dense Detector", "SIFT Detector"])
            
            if detector_type == "Dense Detector":
                step_size = st.slider("Step Size", 10, 50, 20)
                feature_scale = st.slider("Feature Scale", 10, 50, 20)
                img_bound = st.slider("Image Bound", 1, 20, 5)
                
                if st.button("Aplicar Dense Detector", key="dense_detector"):
                    class DenseDetector():
                        def __init__(self, step_size=20, feature_scale=20, img_bound=20):
                            self.initXyStep = step_size
                            self.initFeatureScale = feature_scale
                            self.initImgBound = img_bound
                        
                        def detect(self, img):
                            keypoints = []
                            rows, cols = img.shape[:2]
                            for x in range(self.initImgBound, rows, self.initFeatureScale):
                                for y in range(self.initImgBound, cols, self.initFeatureScale):
                                    keypoints.append(cv2.KeyPoint(float(x), float(y), self.initXyStep))
                            return keypoints
                    
                    dense_detector = DenseDetector(step_size, feature_scale, img_bound)
                    keypoints = dense_detector.detect(img_array)
                    
                    result = img_array.copy()
                    result = cv2.drawKeypoints(result, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, use_container_width=True)
                    st.info(f"Dense Detector encontró {len(keypoints)} características")
                    st.session_state.recognition_result = result_rgb
            
            else:
                n_features = st.slider("Número de características", 0, 1000, 500)
                contrast_threshold = st.slider("Umbral de contraste", 0.01, 0.1, 0.04, 0.01)
                
                if st.button("Aplicar SIFT Detector", key="sift_detector"):
                    class SIFTDetector():
                        def __init__(self, n_features=500, contrast_threshold=0.04):
                            self.detector = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold)
                        
                        def detect(self, img):
                            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            return self.detector.detect(gray_image, None)
                    
                    sift_detector = SIFTDetector(n_features, contrast_threshold)
                    keypoints = sift_detector.detect(img_array)
                    
                    result = img_array.copy()
                    result = cv2.drawKeypoints(result, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, use_container_width=True)
                    st.info(f"SIFT Detector encontró {len(keypoints)} características")
                    st.session_state.recognition_result = result_rgb
        
        st.markdown("---")
        
        if st.button("💾 Descargar resultado", key="download_ch9"):
            if 'recognition_result' in st.session_state:
                pil_image = Image.fromarray(st.session_state.recognition_result)
                
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="⬇️ Descargar imagen con características",
                    data=byte_im,
                    file_name=f"object_recognition_{detector_type.replace(' ', '_')}.png",
                    mime="image/png",
                    use_container_width=True
                )