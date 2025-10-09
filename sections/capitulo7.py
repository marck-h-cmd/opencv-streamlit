import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def show():
    st.header("🔶 Capítulo 7: Shapes and Segmentation")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Sube una imagen", type=['png', 'jpg', 'jpeg'], key="ch7")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        st.session_state.image_loaded = True
    else:
        st.session_state.image_loaded = False
    
    st.subheader("⚙️ Selecciona la operación de formas y segmentación")
    
    col_buttons = st.columns(3)
    with col_buttons[0]:
        if st.button("Contornos", use_container_width=True):
            st.session_state.shape_mode = "Análisis de contornos"
    with col_buttons[1]:
        if st.button("Coincidencia", use_container_width=True):
            st.session_state.shape_mode = "Coincidencia de formas"
    with col_buttons[2]:
        if st.button("Aproximación", use_container_width=True):
            st.session_state.shape_mode = "Aproximación de contornos"
    
    if 'shape_mode' not in st.session_state:
        st.session_state.shape_mode = "Análisis de contornos"
    
    st.markdown("---")
    
    if st.session_state.get('image_loaded', False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📷 Imagen Original**")
            st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with col2:
            st.write(f"**✨ {st.session_state.shape_mode}**")
            
            if st.session_state.shape_mode == "Análisis de contornos":
                threshold1 = st.slider("Umbral 1", 0, 255, 50)
                threshold2 = st.slider("Umbral 2", 0, 255, 150)
                
                if st.button("Analizar Contornos", key="analyze_contours"):
                    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, threshold1, threshold2)
                    
                    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    result = img_array.copy()
                    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
                    
                    for i, contour in enumerate(contours):
                        area = cv2.contourArea(contour)
                        perimeter = cv2.arcLength(contour, True)
                        if area > 100:
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                cv2.putText(result, f'C{i}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, use_container_width=True)
                    st.info(f"✅ Se encontraron {len(contours)} contornos")
                    st.session_state.shape_result = result_rgb
            
            elif st.session_state.shape_mode == "Coincidencia de formas":
                method = st.selectbox("Método de coincidencia", 
                                    ["TM_CCOEFF", "TM_CCOEFF_NORMED", "TM_CCORR", "TM_CCORR_NORMED", "TM_SQDIFF", "TM_SQDIFF_NORMED"])
                
                method_map = {
                    "TM_CCOEFF": cv2.TM_CCOEFF,
                    "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
                    "TM_CCORR": cv2.TM_CCORR,
                    "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
                    "TM_SQDIFF": cv2.TM_SQDIFF,
                    "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED
                }
                
                if st.button("Coincidir Formas", key="match_shapes"):
                    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    
                    template_size = st.slider("Tamaño de plantilla", 10, 100, 30)
                    template = gray[50:50+template_size, 50:50+template_size]
                    
                    result_match = cv2.matchTemplate(gray, template, method_map[method])
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_match)
                    
                    if method in ["TM_SQDIFF", "TM_SQDIFF_NORMED"]:
                        top_left = min_loc
                    else:
                        top_left = max_loc
                    
                    bottom_right = (top_left[0] + template_size, top_left[1] + template_size)
                    
                    result = img_array.copy()
                    cv2.rectangle(result, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.rectangle(result, (50, 50), (50+template_size, 50+template_size), (255, 0, 0), 2)
                    
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, use_container_width=True)
                    st.session_state.shape_result = result_rgb
            
            elif st.session_state.shape_mode == "Aproximación de contornos":
                epsilon_factor = st.slider("Factor epsilon", 0.01, 0.1, 0.03, 0.01)
                
                if st.button("Aproximar Contornos", key="approximate_contours"):
                    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    
                    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    result = img_array.copy()
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 100:
                            epsilon = epsilon_factor * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            
                            cv2.drawContours(result, [approx], -1, (0, 255, 0), 2)
                            
                            if len(approx) == 3:
                                shape_name = "Triángulo"
                            elif len(approx) == 4:
                                shape_name = "Cuadrilátero"
                            elif len(approx) == 5:
                                shape_name = "Pentágono"
                            elif len(approx) == 6:
                                shape_name = "Hexágono"
                            else:
                                shape_name = "Círculo"
                            
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                cv2.putText(result, shape_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, use_container_width=True)
                    st.session_state.shape_result = result_rgb
        
        st.markdown("---")
        
        if st.button("💾 Descargar resultado", key="download_ch7"):
            if 'shape_result' in st.session_state:
                pil_image = Image.fromarray(st.session_state.shape_result)
                
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="⬇️ Descargar imagen procesada",
                    data=byte_im,
                    file_name=f"shapes_{st.session_state.shape_mode.replace(' ', '_')}.png",
                    mime="image/png",
                    use_container_width=True
                )