import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def show():
    st.header("✂️ Capítulo 6: Seam Carving")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Sube una imagen", type=['png', 'jpg', 'jpeg'], key="ch6")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        st.session_state.image_loaded = True
    else:
        st.session_state.image_loaded = False
    
    st.subheader("⚙️ Selecciona la operación de Seam Carving")
    
    col_buttons = st.columns(2)
    with col_buttons[0]:
        if st.button("Expansión", use_container_width=True):
            st.session_state.seam_mode = "Expansión de imágenes"
    with col_buttons[1]:
        if st.button("Eliminación", use_container_width=True):
            st.session_state.seam_mode = "Eliminación de objetos"
    
    if 'seam_mode' not in st.session_state:
        st.session_state.seam_mode = "Expansión de imágenes"
    
    st.markdown("---")
    
    if st.session_state.get('image_loaded', False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📷 Imagen Original**")
            st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with col2:
            st.write(f"**✨ {st.session_state.seam_mode}**")
            
            if st.session_state.seam_mode == "Expansión de imágenes":
                scale_factor = st.slider("Factor de expansión", 1.1, 2.0, 1.3, 0.1)
                
                if st.button("Aplicar Expansión", key="apply_expansion"):
                    original_height, original_width = img_array.shape[:2]
                    new_width = int(original_width * scale_factor)
                    
                    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    
                    energy_map = compute_energy_map(gray)
                    
                    seams = []
                    for i in range(new_width - original_width):
                        seam = find_vertical_seam(energy_map)
                        seams.append(seam)
                        energy_map = remove_seam(energy_map, seam)
                    
                    result = expand_image(img_array, seams)
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, use_container_width=True)
                    st.session_state.seam_result = result_rgb
            
            elif st.session_state.seam_mode == "Eliminación de objetos":
                st.info("Selecciona el área a eliminar")
                st.info("La eliminación puede tardar varios segundos dependiendo del tamaño del área seleccionada")
                
                if 'mask' not in st.session_state:
                    st.session_state.mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
                
                if st.button("Seleccionar área para eliminar", key="select_area"):
                    st.session_state.selecting_area = True
                
                if st.session_state.get('selecting_area', False):
                    st.warning("Haz clic y arrastra en la imagen original para seleccionar el área a eliminar")
                    
                    x = st.slider("Coordenada X", 0, img_array.shape[1]-1, img_array.shape[1]//2)
                    y = st.slider("Coordenada Y", 0, img_array.shape[0]-1, img_array.shape[0]//2)
                    width = st.slider("Ancho del área", 10, 200, 50)
                    height = st.slider("Alto del área", 10, 200, 50)
                    
                    mask_display = img_array.copy()
                    cv2.rectangle(mask_display, (x, y), (x+width, y+height), (0, 0, 255), 2)
                    st.image(cv2.cvtColor(mask_display, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    if st.button("Confirmar selección", key="confirm_selection"):
                        st.session_state.mask[y:y+height, x:x+width] = 1
                        st.session_state.selecting_area = False
                
                if st.button("Eliminar objeto seleccionado", key="remove_object"):
                    if np.sum(st.session_state.mask) > 0:
                        result = remove_object_seam_carving(img_array, st.session_state.mask)
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, use_container_width=True)
                        st.session_state.seam_result = result_rgb
                    else:
                        st.error("Primero selecciona un área para eliminar")
        
        st.markdown("---")
        
        if st.button("💾 Descargar resultado", key="download_ch6"):
            if 'seam_result' in st.session_state:
                pil_image = Image.fromarray(st.session_state.seam_result)
                
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="⬇️ Descargar imagen procesada",
                    data=byte_im,
                    file_name=f"seam_carving_{st.session_state.seam_mode.replace(' ', '_')}.png",
                    mime="image/png",
                    use_container_width=True
                )

def compute_energy_map(img):
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.abs(dx) + np.abs(dy)
    return energy

def find_vertical_seam(energy):
    h, w = energy.shape
    seam = np.zeros(h, dtype=np.int32)
    dist_to = np.copy(energy)
    
    for i in range(1, h):
        for j in range(w):
            if j == 0:
                min_prev = min(dist_to[i-1, j], dist_to[i-1, j+1])
            elif j == w-1:
                min_prev = min(dist_to[i-1, j-1], dist_to[i-1, j])
            else:
                min_prev = min(dist_to[i-1, j-1], dist_to[i-1, j], dist_to[i-1, j+1])
            dist_to[i, j] += min_prev
    
    seam[-1] = np.argmin(dist_to[-1])
    for i in range(h-2, -1, -1):
        j = seam[i+1]
        if j == 0:
            seam[i] = np.argmin(dist_to[i, j:j+2]) + j
        elif j == w-1:
            seam[i] = np.argmin(dist_to[i, j-1:j+1]) + j - 1
        else:
            seam[i] = np.argmin(dist_to[i, j-1:j+2]) + j - 1
    
    return seam

def remove_seam(img, seam):
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        output = np.zeros((h, w-1, 3), dtype=img.dtype)
        for i in range(h):
            j = seam[i]
            output[i] = np.delete(img[i], j, axis=0)
    else:
        output = np.zeros((h, w-1), dtype=img.dtype)
        for i in range(h):
            j = seam[i]
            output[i] = np.delete(img[i], j)
    return output

def expand_image(img, seams):
    h, w = img.shape[:2]
    new_w = w + len(seams)
    
    if len(img.shape) == 3:
        output = np.zeros((h, new_w, 3), dtype=img.dtype)
    else:
        output = np.zeros((h, new_w), dtype=img.dtype)
    
    for i in range(h):
        output[i] = img[i]
        for idx, seam in enumerate(seams):
            j = seam[i]
            output[i] = np.insert(output[i], j, img[i, j], axis=0)
    
    return output

def remove_object_seam_carving(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    energy = compute_energy_map(gray)
    
    energy[mask == 1] -= 1000
    
    h, w = img.shape[:2]
    num_seams = np.sum(mask.any(axis=0))
    
    for _ in range(num_seams):
        seam = find_vertical_seam(energy)
        img = remove_seam(img, seam)
        mask = remove_seam(mask, seam)
        energy = compute_energy_map(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        energy[mask == 1] -= 1000
    
    return img