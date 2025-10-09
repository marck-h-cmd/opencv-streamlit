import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def show():
    st.header("🎯 Capítulo 5: Feature Extraction")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Sube una imagen", type=['png', 'jpg', 'jpeg'], key="ch5")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        st.session_state.image_loaded = True
    else:
        st.session_state.image_loaded = False
    
    st.subheader("⚙️ Selecciona el algoritmo de extracción de características")
    
    col_buttons = st.columns(2)
    with col_buttons[0]:
        if st.button("SIFT", use_container_width=True):
            st.session_state.feature_mode = "SIFT"
    with col_buttons[1]:
        if st.button("FAST", use_container_width=True):
            st.session_state.feature_mode = "FAST"
    
    if 'feature_mode' not in st.session_state:
        st.session_state.feature_mode = "SIFT"
    
    st.markdown("---")
    
    if st.session_state.get('image_loaded', False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📷 Imagen Original**")
            st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with col2:
            st.write(f"**✨ {st.session_state.feature_mode} Features**")
            
            if st.session_state.feature_mode == "SIFT":
                n_features = st.slider("Número de características", 0, 1000, 500)
                n_octave_layers = st.slider("Capas por octava", 1, 10, 3)
                contrast_threshold = st.slider("Umbral de contraste", 0.01, 0.1, 0.04, 0.01)
                edge_threshold = st.slider("Umbral de bordes", 1, 50, 10)
                sigma = st.slider("Sigma", 0.5, 2.0, 1.6, 0.1)
                
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                sift = cv2.SIFT_create(nfeatures=n_features, nOctaveLayers=n_octave_layers, 
                                     contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold, 
                                     sigma=sigma)
                keypoints, descriptors = sift.detectAndCompute(gray, None)
                
                result = cv2.drawKeypoints(img_array, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, use_container_width=True)
                st.info(f"✅ Se detectaron {len(keypoints)} características SIFT")
                st.session_state.feature_result = result_rgb
            
            elif st.session_state.feature_mode == "SURF":
                hessian_threshold = st.slider("Umbral de Hessian", 100, 2000, 400)
                n_octaves = st.slider("Octavas", 1, 10, 4)
                n_octave_layers = st.slider("Capas por octava", 1, 10, 3, key="surf_octave")
                extended = st.checkbox("Descriptores extendidos", value=False)
                upright = st.checkbox("Modo upright", value=False)
                
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                try:
                    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold, 
                                                     nOctaves=n_octaves, 
                                                     nOctaveLayers=n_octave_layers,
                                                     extended=extended, 
                                                     upright=upright)
                    keypoints, descriptors = surf.detectAndCompute(gray, None)
                    
                    result = cv2.drawKeypoints(img_array, keypoints, None, (0, 255, 0), 
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, use_container_width=True)
                    st.info(f"✅ Se detectaron {len(keypoints)} características SURF")
                    st.session_state.feature_result = result_rgb
                except:
                    st.error("SURF no está disponible en esta versión de OpenCV")
                    st.info("Usa SIFT o FAST como alternativas")
            
            elif st.session_state.feature_mode == "FAST":
                threshold = st.slider("Umbral", 1, 100, 10)
                nonmax_suppression = st.checkbox("Supresión no máxima", value=True)
                fast_type = st.selectbox("Tipo FAST", ["TYPE_9_16", "TYPE_7_12", "TYPE_5_8"])
                
                type_map = {
                    "TYPE_9_16": cv2.FAST_FEATURE_DETECTOR_TYPE_9_16,
                    "TYPE_7_12": cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,
                    "TYPE_5_8": cv2.FAST_FEATURE_DETECTOR_TYPE_5_8
                }
                
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                fast = cv2.FastFeatureDetector_create(threshold=threshold, 
                                                    nonmaxSuppression=nonmax_suppression,
                                                    type=type_map[fast_type])
                keypoints = fast.detect(gray, None)
                
                result = cv2.drawKeypoints(img_array, keypoints, None, (0, 0, 255), 
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, use_container_width=True)
                st.info(f"✅ Se detectaron {len(keypoints)} características FAST")
                st.session_state.feature_result = result_rgb
        
        st.markdown("---")
        
        if st.button("💾 Descargar resultado", key="download_ch5"):
            if 'feature_result' in st.session_state:
                pil_image = Image.fromarray(st.session_state.feature_result)
                
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="⬇️ Descargar imagen con características",
                    data=byte_im,
                    file_name=f"features_{st.session_state.feature_mode}.png",
                    mime="image/png",
                    use_container_width=True
                )
    
    else:
        st.info("⬆️ Por favor, sube una imagen para extraer características")
        

        
    
   