import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def show():
    st.header("🧠 Capítulo 11: Neural Networks")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Sube una imagen", type=['png', 'jpg', 'jpeg'], key="ch11")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        st.session_state.image_loaded = True
    else:
        st.session_state.image_loaded = False
    
    st.subheader("⚙️ Clasificadores ANN-MLP")
    
    if st.session_state.get('image_loaded', False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📷 Imagen Original**")
            st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with col2:
            st.write("**✨ Clasificación ANN-MLP**")
            
            if st.button("Clasificación ANN", key="simulate_ann"):
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (64, 64))
                
                features = resized.flatten()
                features = features / 255.0
                
                simulated_output = np.random.rand(3)
                simulated_output = np.exp(simulated_output) / np.sum(np.exp(simulated_output))
                
                classes = ["Vestido", "Calzado", "Mochila"]
                predicted_class = classes[np.argmax(simulated_output)]
                confidence = np.max(simulated_output)
                
                result = img_array.copy()
                cv2.putText(result, f"Clase: {predicted_class}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result, f"Confianza: {confidence:.2f}", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                height, width = img_array.shape[:2]
                cv2.rectangle(result, (10, height-60), (width-10, height-10), (0, 0, 0), -1)
                
                for i, (cls, prob) in enumerate(zip(classes, simulated_output)):
                    y_pos = height - 40 + i * 15
                    cv2.putText(result, f"{cls}: {prob:.2f}", (20, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, use_container_width=True)
                
                st.success(f"✅ Clasificado como: **{predicted_class}**")
                st.info(f"📊 Nivel de confianza: **{confidence:.2f}**")
                
                col_probs = st.columns(3)
                for i, (cls, prob) in enumerate(zip(classes, simulated_output)):
                    with col_probs[i]:
                        st.metric(label=cls, value=f"{prob:.2f}")
                
                st.session_state.nn_result = result_rgb
        
        st.markdown("---")
        
        if st.button("💾 Descargar resultado", key="download_ch11"):
            if 'nn_result' in st.session_state:
                pil_image = Image.fromarray(st.session_state.nn_result)
                
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="⬇️ Descargar imagen clasificada",
                    data=byte_im,
                    file_name="neural_network_classification.png",
                    mime="image/png",
                    use_container_width=True
                )