import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def show():
    st.header("🔍 Capítulo 2: Edges and Filters")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Sube una imagen", type=['png', 'jpg', 'jpeg'], key="ch2")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        st.subheader("⚙️ Selecciona un filtro o efecto")
        
        col_buttons = st.columns(4)
        with col_buttons[0]:
            if st.button("Convolución 2D", use_container_width=True):
                st.session_state.filter_type = "Convolución 2D"
        with col_buttons[1]:
            if st.button("Desenfoque", use_container_width=True):
                st.session_state.filter_type = "Desenfoque (Blurring)"
        with col_buttons[2]:
            if st.button("Motion Blur", use_container_width=True):
                st.session_state.filter_type = "Motion blur"
        with col_buttons[3]:
            if st.button("Enfoque", use_container_width=True):
                st.session_state.filter_type = "Enfoque (Sharpening)"
        
        if 'filter_type' not in st.session_state:
            st.session_state.filter_type = "Convolución 2D"
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📷 Imagen Original**")
            st.image(image, use_container_width=True)
        
        with col2:
            st.write("**✨ Resultado**")
            
            if st.session_state.filter_type == "Convolución 2D":
                kernel_type = st.selectbox(
                    "Tipo de kernel:",
                    ["Identidad", "Box Blur", "Sharpen", "Emboss", "Outline", "Sobel X", "Sobel Y", "Laplacian"]
                )
                
                if kernel_type == "Identidad":
                    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
                elif kernel_type == "Box Blur":
                    kernel = np.ones((3, 3), np.float32) / 9
                elif kernel_type == "Sharpen":
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                elif kernel_type == "Emboss":
                    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
                elif kernel_type == "Outline":
                    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                elif kernel_type == "Sobel X":
                    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                elif kernel_type == "Sobel Y":
                    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                elif kernel_type == "Laplacian":
                    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
                
                result = cv2.filter2D(img_array, -1, kernel)
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) if len(result.shape) == 3 else result
                st.image(result_rgb, use_container_width=True)
                
                st.write("**Kernel aplicado:**")
                st.dataframe(kernel)
            
            elif st.session_state.filter_type == "Desenfoque (Blurring)":
                blur_type = st.selectbox(
                    "Tipo de desenfoque:",
                    ["Gaussian Blur", "Median Blur", "Bilateral Filter"]
                )
                
                if blur_type == "Gaussian Blur":
                    kernel_size = st.slider("Tamaño del kernel", 3, 15, 5, 2)
                    sigma = st.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
                    result = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), sigma)
                
                elif blur_type == "Median Blur":
                    kernel_size = st.slider("Tamaño del kernel", 3, 15, 5, 2)
                    result = cv2.medianBlur(img_array, kernel_size)
                
                else:
                    d = st.slider("Diámetro", 1, 15, 9, 2)
                    sigma_color = st.slider("Sigma Color", 1.0, 100.0, 75.0, 1.0)
                    sigma_space = st.slider("Sigma Space", 1.0, 100.0, 75.0, 1.0)
                    result = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)
                
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) if len(result.shape) == 3 else result
                st.image(result_rgb, use_container_width=True)
            
            elif st.session_state.filter_type == "Motion blur":
                kernel_size = st.slider("Tamaño del kernel", 3, 30, 15, 2)
                angle = st.slider("Ángulo de movimiento", 0, 180, 0, 1)
                
                kernel = np.zeros((kernel_size, kernel_size))
                center = kernel_size // 2
                
                if angle == 0 or angle == 180:
                    kernel[center, :] = 1
                elif angle == 90:
                    kernel[:, center] = 1
                else:
                    radians = np.radians(angle)
                    for i in range(kernel_size):
                        x = int(center + (i - center) * np.cos(radians))
                        y = int(center + (i - center) * np.sin(radians))
                        if 0 <= x < kernel_size and 0 <= y < kernel_size:
                            kernel[y, x] = 1
                
                kernel = kernel / np.sum(kernel)
                result = cv2.filter2D(img_array, -1, kernel)
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) if len(result.shape) == 3 else result
                st.image(result_rgb, use_container_width=True)
            
            elif st.session_state.filter_type == "Enfoque (Sharpening)":
                sharpen_type = st.selectbox(
                    "Método de enfoque:",
                    ["Filtro Laplaciano", "Unsharp Mask", "High Pass Filter"]
                )
                
                if sharpen_type == "Filtro Laplaciano":
                    strength = st.slider("Fuerza de enfoque", 0.1, 3.0, 1.0, 0.1)
                    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) * strength
                    kernel[1, 1] += 1
                    result = cv2.filter2D(img_array, -1, kernel)
                
                elif sharpen_type == "Unsharp Mask":
                    blur_amount = st.slider("Cantidad de desenfoque", 1.0, 5.0, 2.0, 0.1)
                    strength = st.slider("Fuerza", 0.1, 3.0, 1.5, 0.1)
                    blurred = cv2.GaussianBlur(img_array, (0, 0), blur_amount)
                    result = cv2.addWeighted(img_array, 1.0 + strength, blurred, -strength, 0)
                
                else:
                    kernel_size = st.slider("Tamaño del kernel", 3, 15, 5, 2)
                    low_pass = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
                    result = cv2.subtract(img_array, low_pass)
                    result = cv2.add(img_array, result)
                
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) if len(result.shape) == 3 else result
                st.image(result_rgb, use_container_width=True)
        
        st.markdown("---")
        
        result_for_download = result_rgb if 'result_rgb' in locals() else cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        if st.button("💾 Descargar resultado", key="download_ch2"):
            pil_image = Image.fromarray(result_for_download)
            
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="⬇️ Descargar imagen procesada",
                data=byte_im,
                file_name=f"processed_{st.session_state.filter_type.lower().replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True
            )
    
    else:
        st.info("⬆️ Por favor, sube una imagen para comenzar a aplicar filtros")
        
        st.markdown("---")
        st.subheader("📖 Filtros disponibles:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - **Convolución 2D**: Aplica kernels personalizados
            - **Desenfoque Gaussiano**: Suavizado con distribución gaussiana
            - **Desenfoque Mediano**: Elimina ruido preservando bordes
            - **Filtro Bilateral**: Preserva bordes mientras suaviza
            """)
        
        with col2:
            st.markdown("""
            - **Motion Blur**: Simula movimiento de cámara
            - **Enfoque Laplaciano**: Realza bordes y detalles
            - **Unsharp Mask**: Enfoque profesional
            - **High Pass Filter**: Enfatiza detalles finos
            """)