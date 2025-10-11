import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="OpenCV 3.x with Python By Example Second Edition",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar session state
if 'selected_capitulo' not in st.session_state:
    st.session_state.selected_capitulo = None

# Título principal
st.title("📷 OpenCV 3.x with Python By Example Second Edition")
st.markdown("---")

# Sidebar con botones de capítulos
st.sidebar.title("📚 Capítulos")
st.sidebar.markdown("Selecciona un capítulo")
st.sidebar.markdown("---")

# Definir capítulos
capitulos = [
    {"num": 1, "nombre": "Geometric Transformations", "icon": "🔄"},
    {"num": 2, "nombre": "Edges and Filters", "icon": "🔍"},
    {"num": 3, "nombre": "Cartoonizing Images", "icon": "🎨"},
    {"num": 4, "nombre": "Body Parts Detection", "icon": "👤"},
    {"num": 5, "nombre": "Feature Extraction", "icon": "🎯"},
    {"num": 6, "nombre": "Seam Carving", "icon": "✂️"},
    {"num": 7, "nombre": "Shapes and Segmentation", "icon": "🔶"},
    {"num": 8, "nombre": "Object Tracking", "icon": "🎯"},
    {"num": 9, "nombre": "Object Recognition", "icon": "🤖"},
    {"num": 10, "nombre": "Augmented Reality", "icon": "🌐"},
    {"num": 11, "nombre": "Neural Networks", "icon": "🧠"}
]

# Crear botones para cada capítulo en el sidebar
for cap in capitulos:
    if st.sidebar.button(
        f"{cap['icon']} Cap {cap['num']}: {cap['nombre']}", 
        key=f"cap_{cap['num']}",
        use_container_width=True
    ):
        st.session_state.selected_capitulo = cap['num']

st.sidebar.markdown("---")
st.sidebar.info("💡 Selecciona un capítulo para ver sus algoritmos")

# Función para mostrar mensaje de módulo pendiente
def show_pending_module(capitulo_num, capitulo_name):
    st.info(f"📦 El módulo `capitulo_{capitulo_num}.py` aún no está implementado")
    st.write(f"""
    Para implementar este capítulo, crea el archivo `capitulo_{capitulo_num}.py` con la siguiente estructura:
    """)
    
    st.code(f'''
import streamlit as st
import cv2
import numpy as np
from PIL import Image

def show():
    st.header("Capítulo {capitulo_num}: {capitulo_name}")
    
    # Subir imagen
    uploaded_file = st.file_uploader("Sube una imagen", type=['png', 'jpg', 'jpeg'], key="ch{capitulo_num}")
    
    if uploaded_file is not None:
        # Leer imagen
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Convertir a BGR para OpenCV
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Crear pestañas para cada algoritmo
        tabs = st.tabs([
            "Algoritmo 1",
            "Algoritmo 2", 
            "Algoritmo 3",
            "Algoritmo 4"
        ])
        
        with tabs[0]:
            st.subheader("Algoritmo 1")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Imagen Original**")
                st.image(image, use_container_width=True)
            
            with col2:
                st.write("**Resultado**")
                # Aplicar algoritmo aquí
                result = img_array.copy()
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, use_container_width=True)
            
            # Parámetros
            st.markdown("---")
            st.subheader("⚙️ Parámetros")
            param1 = st.slider("Parámetro 1", 0, 100, 50)
        
        # Repetir para otras pestañas...
    else:
        st.warning("⬆️ Por favor, sube una imagen para comenzar")
''', language='python')

# Contenido principal según el capítulo seleccionado
if st.session_state.selected_capitulo is not None:
    capitulo_num = st.session_state.selected_capitulo
    
    try:
        # Intentar importar el módulo del capítulo
        if capitulo_num == 1:
            from sections import capitulo1
            capitulo1.show()
        elif capitulo_num == 2:
            from sections import capitulo2
            capitulo2.show()
        elif capitulo_num == 3:
            from sections import capitulo3
            capitulo3.show()
        elif capitulo_num == 4:
            from sections import capitulo4
            capitulo4.show()
        elif capitulo_num == 5:
            from sections import capitulo5
            capitulo5.show()
        elif capitulo_num == 6:
            from sections import capitulo6
            capitulo6.show()
        elif capitulo_num == 7:
            from sections import capitulo7
            capitulo7.show()
        elif capitulo_num == 8:
            from sections import capitulo8
            capitulo8.show()
        elif capitulo_num == 9:
            from sections import capitulo9
            capitulo9.show()
        elif capitulo_num == 10:
            from sections import capitulo10
            capitulo10.show()
        elif capitulo_num == 11:
            from sections import capitulo11
            capitulo11.show()
    
    except ImportError:
        # Mostrar mensaje si el módulo no existe
        cap_info = capitulos[capitulo_num - 1]
        show_pending_module(capitulo_num, cap_info['nombre'])

else:

    st.info("👈 Selecciona un capítulo del menú lateral para comenzar")
    
    st.subheader("📘 Sobre este Dashboard")
    st.write("""
    ## Alumno: Hermenegildo Pacheco Marck Alessandro
    
    ## Cada capítulo contiene algoritmos para aplicar  a imágenes.
    
    ### Nota: Es posible que algunos capítulos no permita el accesso a la cámara web debido a que esta en entorno de Streamlit Cloud.
    """)
    
  
 

 

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p> OpenCV 3.x with Python By Example Second Edition</p>
    </div>
    """,
    unsafe_allow_html=True
)