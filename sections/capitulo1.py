import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def show():
    st.header("🔄 Capítulo 1: Transformaciones Geométricas")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Sube una imagen", type=['png', 'jpg', 'jpeg'], key="ch1")
    
    img_array = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        st.session_state.image_loaded = True
    else:
        st.session_state.image_loaded = False
    
    st.subheader("⚙️ Selecciona una transformación")
    
    col_buttons = st.columns(4)
    with col_buttons[0]:
        if st.button("Canales", use_container_width=True):
            st.session_state.transformation = "División y combinación de canales"
    with col_buttons[1]:
        if st.button("Traslación", use_container_width=True):
            st.session_state.transformation = "Traslación de imágenes"
    with col_buttons[2]:
        if st.button("Rotación", use_container_width=True):
            st.session_state.transformation = "Rotación de imágenes"
    with col_buttons[3]:
        if st.button("Escalado", use_container_width=True):
            st.session_state.transformation = "Escalado de imágenes"
    
    col_buttons2 = st.columns(3)
    with col_buttons2[0]:
        if st.button("Afines", use_container_width=True):
            st.session_state.transformation = "Transformaciones afines"
    with col_buttons2[1]:
        if st.button("Proyectivas", use_container_width=True):
            st.session_state.transformation = "Transformaciones proyectivas"
    with col_buttons2[2]:
        if st.button("Deformación", use_container_width=True):
            st.session_state.transformation = "Deformación de imágenes"
    
    if 'transformation' not in st.session_state:
        st.session_state.transformation = "División y combinación de canales"
    
    st.markdown("---")
    
    if st.session_state.get('image_loaded', False) and img_array is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📷 Imagen Original**")
            st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with col2:
            st.write(f"**✨ {st.session_state.transformation}**")
            
            if st.session_state.transformation == "División y combinación de canales":
                if len(img_array.shape) == 3:
                    b, g, r = cv2.split(img_array)
                    
                    canal_option = st.selectbox("Selecciona visualización", 
                                               ["Imagen Original", "Solo Canal Rojo", "Solo Canal Verde", "Solo Canal Azul"])
                    
                    if canal_option == "Imagen Original":
                        result_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, use_container_width=True)
                    elif canal_option == "Solo Canal Rojo":
                        zeros = np.zeros_like(r)
                        result = cv2.merge([zeros, zeros, r])
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, use_container_width=True)
                    elif canal_option == "Solo Canal Verde":
                        zeros = np.zeros_like(g)
                        result = cv2.merge([zeros, g, zeros])
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, use_container_width=True)
                    elif canal_option == "Solo Canal Azul":
                        zeros = np.zeros_like(b)
                        result = cv2.merge([b, zeros, zeros])
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, use_container_width=True)
                    
                    st.session_state.transform_result = result_rgb
                else:
                    st.info("La imagen es en escala de grises (1 canal)")
                    result_rgb = img_array
                    st.image(result_rgb, use_container_width=True)
                    st.session_state.transform_result = result_rgb
            
            elif st.session_state.transformation == "Traslación de imágenes":
                tx = st.slider("Desplazamiento X (píxeles)", -200, 200, 50, key="tx")
                ty = st.slider("Desplazamiento Y (píxeles)", -200, 200, 30, key="ty")
                
                rows, cols = img_array.shape[:2]
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                translated = cv2.warpAffine(img_array, M, (cols, rows))
                
                result_rgb = cv2.cvtColor(translated, cv2.COLOR_BGR2RGB) if len(translated.shape) == 3 else translated
                st.image(result_rgb, use_container_width=True)
                st.session_state.transform_result = result_rgb
            
            elif st.session_state.transformation == "Rotación de imágenes":
                angle = st.slider("Ángulo de rotación (grados)", -180, 180, 45, key="angle")
                scale = st.slider("Escala", 0.1, 2.0, 1.0, 0.1, key="rot_scale")
                
                rows, cols = img_array.shape[:2]
                center = (cols // 2, rows // 2)
                M = cv2.getRotationMatrix2D(center, angle, scale)
                rotated = cv2.warpAffine(img_array, M, (cols, rows))
                
                result_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB) if len(rotated.shape) == 3 else rotated
                st.image(result_rgb, use_container_width=True)
                st.session_state.transform_result = result_rgb
            
            elif st.session_state.transformation == "Escalado de imágenes":
                scale_x = st.slider("Escala X", 0.1, 3.0, 1.5, 0.1, key="scale_x")
                scale_y = st.slider("Escala Y", 0.1, 3.0, 1.5, 0.1, key="scale_y")
                
                interpolation_method = st.selectbox(
                    "Método de interpolación",
                    ["INTER_LINEAR", "INTER_CUBIC", "INTER_NEAREST", "INTER_AREA"]
                )
                
                interp_map = {
                    "INTER_LINEAR": cv2.INTER_LINEAR,
                    "INTER_CUBIC": cv2.INTER_CUBIC,
                    "INTER_NEAREST": cv2.INTER_NEAREST,
                    "INTER_AREA": cv2.INTER_AREA
                }
                
                scaled = cv2.resize(img_array, None, fx=scale_x, fy=scale_y, 
                                   interpolation=interp_map[interpolation_method])
                
                result_rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB) if len(scaled.shape) == 3 else scaled
                st.image(result_rgb, use_container_width=True)
                st.session_state.transform_result = result_rgb
            
            elif st.session_state.transformation == "Transformaciones afines":
                rows, cols = img_array.shape[:2]
                
                transform_type = st.radio(
                    "Tipo de transformación afín:",
                    ["Inclinación horizontal (Shear)", "Inclinación vertical", "Personalizada"],
                    key="affine_type"
                )
                
                if transform_type == "Inclinación horizontal (Shear)":
                    shear_factor = st.slider("Factor de inclinación", 0.0, 1.0, 0.3, 0.1)
                    pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
                    pts2 = np.float32([[0, 0], [cols-1, 0], [int(shear_factor*cols), rows-1]])
                
                elif transform_type == "Inclinación vertical":
                    shear_factor = st.slider("Factor de inclinación", 0.0, 1.0, 0.3, 0.1)
                    pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
                    pts2 = np.float32([[0, 0], [cols-1, int(shear_factor*rows)], [0, rows-1]])
                
                else:
                    st.write("Ajusta los puntos de transformación:")
                    offset_x = st.slider("Desplazamiento X inferior", -200, 200, 50, key="aff_x")
                    offset_y = st.slider("Desplazamiento Y derecho", -200, 200, 30, key="aff_y")
                    
                    pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
                    pts2 = np.float32([[0, 0], [cols-1, offset_y], [offset_x, rows-1]])
                
                M = cv2.getAffineTransform(pts1, pts2)
                affine = cv2.warpAffine(img_array, M, (cols, rows))
                
                result_rgb = cv2.cvtColor(affine, cv2.COLOR_BGR2RGB) if len(affine.shape) == 3 else affine
                st.image(result_rgb, use_container_width=True)
                st.session_state.transform_result = result_rgb
            
            elif st.session_state.transformation == "Transformaciones proyectivas":
                rows, cols = img_array.shape[:2]
                
                preset = st.selectbox(
                    "Preset de perspectiva:",
                    ["Vista desde arriba", "Vista lateral izquierda", "Vista lateral derecha", "Personalizado"]
                )
                
                if preset == "Vista desde arriba":
                    factor = st.slider("Intensidad", 0.0, 0.5, 0.2, 0.05)
                    pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
                    pts2 = np.float32([[cols*factor, 0], [cols*(1-factor), 0], [0, rows], [cols, rows]])
                
                elif preset == "Vista lateral izquierda":
                    factor = st.slider("Intensidad", 0.0, 0.5, 0.2, 0.05)
                    pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
                    pts2 = np.float32([[0, rows*factor], [cols, 0], [0, rows*(1-factor)], [cols, rows]])
                
                elif preset == "Vista lateral derecha":
                    factor = st.slider("Intensidad", 0.0, 0.5, 0.2, 0.05)
                    pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
                    pts2 = np.float32([[0, 0], [cols, rows*factor], [0, rows], [cols, rows*(1-factor)]])
                
                else:
                    st.write("Ajusta los puntos de las esquinas:")
                    offset = st.slider("Offset de perspectiva", 0, 200, 100)
                    pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
                    pts2 = np.float32([
                        [offset, offset], 
                        [cols-offset, offset], 
                        [0, rows], 
                        [cols, rows]
                    ])
                
                M = cv2.getPerspectiveTransform(pts1, pts2)
                perspective = cv2.warpPerspective(img_array, M, (cols, rows))
                
                result_rgb = cv2.cvtColor(perspective, cv2.COLOR_BGR2RGB) if len(perspective.shape) == 3 else perspective
                st.image(result_rgb, use_container_width=True)
                st.session_state.transform_result = result_rgb
            
            elif st.session_state.transformation == "Deformación de imágenes":
                rows, cols = img_array.shape[:2]
                
                deform_type = st.selectbox(
                    "Tipo de deformación:",
                    ["Onda horizontal", "Onda vertical", "Remolino", "Esfera"]
                )
                
                if deform_type == "Onda horizontal":
                    amplitude = st.slider("Amplitud", 0, 50, 20)
                    frequency = st.slider("Frecuencia", 0.01, 0.1, 0.05, 0.01)
                    
                    map_x = np.zeros((rows, cols), dtype=np.float32)
                    map_y = np.zeros((rows, cols), dtype=np.float32)
                    
                    for i in range(rows):
                        for j in range(cols):
                            map_x[i, j] = j
                            map_y[i, j] = i + amplitude * np.sin(j * frequency)
                    
                    warped = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR)
                
                elif deform_type == "Onda vertical":
                    amplitude = st.slider("Amplitud", 0, 50, 20)
                    frequency = st.slider("Frecuencia", 0.01, 0.1, 0.05, 0.01)
                    
                    map_x = np.zeros((rows, cols), dtype=np.float32)
                    map_y = np.zeros((rows, cols), dtype=np.float32)
                    
                    for i in range(rows):
                        for j in range(cols):
                            map_x[i, j] = j + amplitude * np.sin(i * frequency)
                            map_y[i, j] = i
                    
                    warped = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR)
                
                elif deform_type == "Remolino":
                    strength = st.slider("Intensidad", 0.0, 2.0, 0.5, 0.1)
                    
                    center_x, center_y = cols // 2, rows // 2
                    map_x = np.zeros((rows, cols), dtype=np.float32)
                    map_y = np.zeros((rows, cols), dtype=np.float32)
                    
                    for i in range(rows):
                        for j in range(cols):
                            dx = j - center_x
                            dy = i - center_y
                            distance = np.sqrt(dx**2 + dy**2)
                            angle = np.arctan2(dy, dx) + strength * distance / 100
                            map_x[i, j] = center_x + distance * np.cos(angle)
                            map_y[i, j] = center_y + distance * np.sin(angle)
                    
                    warped = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR)
                
                else:  
                    strength = st.slider("Intensidad", 0.0, 2.0, 1.0, 0.1)
                    
                    center_x, center_y = cols // 2, rows // 2
                    radius = min(center_x, center_y)
                    map_x = np.zeros((rows, cols), dtype=np.float32)
                    map_y = np.zeros((rows, cols), dtype=np.float32)
                    
                    for i in range(rows):
                        for j in range(cols):
                            dx = (j - center_x) / radius
                            dy = (i - center_y) / radius
                            distance = min(np.sqrt(dx**2 + dy**2), 1.0)
                            factor = (1.0 - distance**2) * strength
                            map_x[i, j] = j - dx * factor * radius
                            map_y[i, j] = i - dy * factor * radius
                    
                    warped = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR)
                
                result_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB) if len(warped.shape) == 3 else warped
                st.image(result_rgb, use_container_width=True)
                st.session_state.transform_result = result_rgb
        
        st.markdown("---")
        
        if st.button("💾 Descargar resultado", key="download_ch1"):
            if 'transform_result' in st.session_state:
                pil_image = Image.fromarray(st.session_state.transform_result)
                
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="⬇️ Descargar imagen transformada",
                    data=byte_im,
                    file_name=f"transformacion_{st.session_state.transformation.replace(' ', '_')}.png",
                    mime="image/png",
                    use_container_width=True
                )
    
    else:
        st.info("⬆️ Por favor, sube una imagen para comenzar a aplicar transformaciones")