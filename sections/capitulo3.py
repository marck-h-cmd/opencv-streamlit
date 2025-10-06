import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile

def show():
    st.header("🎨 Capítulo 3: Cartoonizing Images")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Sube una imagen o video", type=['png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'], key="ch3")
    
    if uploaded_file is not None:
        file_type = uploaded_file.type
        if 'image' in file_type:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            st.session_state.image_loaded = True
            st.session_state.video_loaded = False
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name
            st.session_state.video_loaded = True
            st.session_state.image_loaded = False
    
    st.subheader("⚙️ Selecciona una funcionalidad")
    
    col_buttons = st.columns(3)
    with col_buttons[0]:
        if st.button("Webcam/Video", use_container_width=True):
            st.session_state.cartoon_mode = "Webcam y Video"
    with col_buttons[1]:
        if st.button("Mouse", use_container_width=True):
            st.session_state.cartoon_mode = "Entradas de mouse"
    with col_buttons[2]:
        if st.button("Cartoon", use_container_width=True):
            st.session_state.cartoon_mode = "Efecto cartoon"
    
    if 'cartoon_mode' not in st.session_state:
        st.session_state.cartoon_mode = "Efecto cartoon"
    
    st.markdown("---")
    
    if st.session_state.cartoon_mode == "Webcam y Video":
        st.info("🔴🎥 Webcam y Procesamiento de Video con Interacción")
        
        option = st.radio("Selecciona fuente:", ["Cámara Web", "Archivo de Video"])
        
        if option == "Cámara Web":
            if st.button("Iniciar Webcam Interactiva", key="start_interactive_webcam"):
                st.session_state.interactive_webcam = True
            
            if st.button("Detener Webcam", key="stop_interactive_webcam"):
                st.session_state.interactive_webcam = False
            
            if st.session_state.get('interactive_webcam', False):
                st.info("🖱️ Haz clic y arrastra para dibujar rectángulos invertidos en la webcam")
                
                cap = cv2.VideoCapture(0)
                
                if cap.isOpened():
                    stframe = st.empty()
                    stop_button = st.button("🛑 Detener Webcam")
                    
                    drawing = False
                    x_init, y_init = -1, -1
                    event_params = {"top_left_pt": (-1, -1), "bottom_right_pt": (-1, -1)}
                    
                    def update_pts(params, x, y):
                        global x_init, y_init
                        params["top_left_pt"] = (min(x_init, x), min(y_init, y))
                        params["bottom_right_pt"] = (max(x_init, x), max(y_init, y))
                    
                    while cap.isOpened() and not stop_button:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        img = cv2.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
                        
                        (x0,y0), (x1,y1) = event_params["top_left_pt"], event_params["bottom_right_pt"]
                        if x0 != -1 and y0 != -1 and x1 != -1 and y1 != -1:
                            img[y0:y1, x0:x1] = 255 - img[y0:y1, x0:x1]
                        
                        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                        if stop_button:
                            break
                    
                    cap.release()
                else:
                    st.error("No se pudo acceder a la cámara web")
        
        else:
            if st.session_state.get('video_loaded', False):
                if st.button("Procesar Video Interactivo", key="process_interactive_video"):
                    st.session_state.process_video = True
                
                if st.session_state.get('process_video', False):
                    cap = cv2.VideoCapture(st.session_state.video_path)
                    
                    if cap.isOpened():
                        stframe = st.empty()
                        stop_button = st.button("🛑 Detener Video")
                        
                        drawing = False
                        x_init, y_init = -1, -1
                        event_params = {"top_left_pt": (-1, -1), "bottom_right_pt": (-1, -1)}
                        
                        while cap.isOpened() and not stop_button:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            img = cv2.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
                            
                            (x0,y0), (x1,y1) = event_params["top_left_pt"], event_params["bottom_right_pt"]
                            if x0 != -1 and y0 != -1 and x1 != -1 and y1 != -1:
                                img[y0:y1, x0:x1] = 255 - img[y0:y1, x0:x1]
                            
                            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                        cap.release()
                    else:
                        st.error("No se pudo abrir el video")
            else:
                st.info("⬆️ Por favor, sube un video para procesar")
        
        code_demo = """
import cv2
import numpy as np

def update_pts(params, x, y):
    global x_init, y_init
    params["top_left_pt"] = (min(x_init, x), min(y_init, y))
    params["bottom_right_pt"] = (max(x_init, x), max(y_init, y))
    img[y_init:y, x_init:x] = 255 - img[y_init:y, x_init:x]

def draw_rectangle(event, x, y, flags, params):
    global x_init, y_init, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_init, y_init = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        update_pts(params, x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        update_pts(params, x, y)

if __name__=='__main__':
    drawing = False
    event_params = {"top_left_pt": (-1, -1), "bottom_right_pt": (-1, -1)}
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    cv2.namedWindow('Webcam')
    cv2.setMouseCallback('Webcam', draw_rectangle, event_params)
    
    while True:
        ret, frame = cap.read()
        img = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        (x0,y0), (x1,y1) = event_params["top_left_pt"], event_params["bottom_right_pt"]
        img[y0:y1, x0:x1] = 255 - img[y0:y1, x0:x1]
        cv2.imshow('Webcam', img)
        c = cv2.waitKey(1)
        if c == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
        """
        with st.expander("💻 Código para webcam/video interactivo"):
            st.code(code_demo, language="python")
    
    elif st.session_state.cartoon_mode == "Entradas de mouse":
        st.info("🖱️ Funcionalidad de Entradas de Mouse")
        
        if st.session_state.get('image_loaded', False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📷 Imagen Original**")
                st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with col2:
                st.write("**✨ Imagen con Interacción**")
                
                mouse_action = st.selectbox("Acción del mouse:",
                                          ["Dibujar círculos", "Seleccionar región", "Dibujar rectángulos"])
                
                if st.button("Aplicar efecto mouse", key="apply_mouse"):
                    result = img_array.copy()
                    
                    if mouse_action == "Dibujar círculos":
                        cv2.circle(result, (100, 100), 30, (0, 255, 0), -1)
                        cv2.circle(result, (200, 150), 25, (255, 0, 0), -1)
                    elif mouse_action == "Dibujar rectángulos":
                        cv2.rectangle(result, (50, 50), (150, 150), (0, 0, 255), 3)
                        cv2.rectangle(result, (180, 80), (280, 180), (255, 255, 0), 2)
                    else:
                        cv2.rectangle(result, (75, 75), (225, 225), (255, 0, 255), 2)
                    
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, use_container_width=True)
                    st.session_state.mouse_result = result_rgb
        
        else:
            st.info("⬆️ Por favor, sube una imagen para interactuar")
        
        code_demo = """
import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.rectangle(img, (x-15, y-15), (x+15, y+15), (255, 0, 0), 2)

img = np.zeros((500, 500, 3), np.uint8)
cv2.namedWindow('Mouse Interactions')
cv2.setMouseCallback('Mouse Interactions', mouse_callback)

while True:
    cv2.imshow('Mouse Interactions', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
        """
        with st.expander("💻 Código para entradas de mouse"):
            st.code(code_demo, language="python")
    
    elif st.session_state.cartoon_mode == "Efecto cartoon":
        st.info("🎨 Efecto Cartoon en Imagen")
        
        if st.session_state.get('image_loaded', False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📷 Imagen Original**")
                st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with col2:
                st.write("**✨ Resultado Cartoon**")
                
                cartoon_style = st.selectbox("Estilo cartoon:",
                                           ["Clásico", "Acuarela", "Cómic", "Sketch"])
                
                if cartoon_style == "Clásico":
                    line_size = st.slider("Tamaño de líneas", 1, 15, 7, key="line_size")
                    blur_value = st.slider("Valor de desenfoque", 1, 15, 7, key="blur_value")
                    total_color = st.slider("Reducción de color", 2, 20, 8, key="total_color")
                    
                    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    gray_blur = cv2.medianBlur(gray, blur_value)
                    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
                    
                    data = np.float32(img_array).reshape((-1, 3))
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
                    ret, label, center = cv2.kmeans(data, total_color, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    center = np.uint8(center)
                    result = center[label.flatten()]
                    result = result.reshape(img_array.shape)
                    
                    blurred = cv2.bilateralFilter(result, d=7, sigmaColor=200, sigmaSpace=200)
                    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
                    result_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
                
                elif cartoon_style == "Acuarela":
                    blur_value = st.slider("Intensidad acuarela", 1, 15, 10, key="watercolor")
                    result = cv2.stylization(img_array, sigma_s=60, sigma_r=0.45)
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                
                elif cartoon_style == "Cómic":
                    line_size = st.slider("Grosor de líneas", 1, 15, 5, key="comic_lines")
                    total_color = st.slider("Colores cómic", 2, 10, 4, key="comic_colors")
                    
                    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    gray_blur = cv2.medianBlur(gray, 7)
                    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, 7)
                    
                    data = np.float32(img_array).reshape((-1, 3))
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
                    ret, label, center = cv2.kmeans(data, total_color, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    center = np.uint8(center)
                    result = center[label.flatten()]
                    result = result.reshape(img_array.shape)
                    
                    cartoon = cv2.bitwise_and(result, result, mask=edges)
                    result_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
                
                else:
                    pencil_style = st.radio("Estilo de sketch:", ["Color", "Blanco y negro"])
                    
                    if pencil_style == "Color":
                        result, _ = cv2.pencilSketch(img_array, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    else:
                        result, _ = cv2.pencilSketch(img_array, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
                        result_rgb = result
                
                st.image(result_rgb, use_container_width=True)
                st.session_state.cartoon_result = result_rgb
        
        else:
            st.info("⬆️ Por favor, sube una imagen para aplicar efecto cartoon")
    
    st.markdown("---")
    
    if st.button("💾 Descargar resultado", key="download_ch3"):
        result_for_download = None
        
        if st.session_state.cartoon_mode == "Entradas de mouse" and 'mouse_result' in st.session_state:
            result_for_download = st.session_state.mouse_result
        elif st.session_state.cartoon_mode == "Efecto cartoon" and 'cartoon_result' in st.session_state:
            result_for_download = st.session_state.cartoon_result
        elif st.session_state.get('image_loaded', False):
            result_for_download = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        if result_for_download is not None:
            pil_image = Image.fromarray(result_for_download)
            
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="⬇️ Descargar imagen procesada",
                data=byte_im,
                file_name=f"cartoon_{st.session_state.cartoon_mode.lower().replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.warning("No hay resultado para descargar")
    
    else:
        st.info("⬆️ Por favor, sube una imagen o video para comenzar")
        
        st.markdown("---")
        st.subheader("📖 Funcionalidades disponibles:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - **Webcam y Video**: Captura interactiva con rectángulos invertidos
            - **Entradas de mouse**: Interacción con clics y movimiento
            - **Procesamiento real**: Funcionalidades operativas en tiempo real
            """)
        
        with col2:
            st.markdown("""
            - **Efecto cartoon**: Transformación a estilo animado
            - **Estilos variados**: Clásico, acuarela, cómic, sketch
            - **Interacción visual**: Dibujo directo en video/webcam
            """)