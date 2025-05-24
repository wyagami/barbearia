import streamlit as st
import os
import tempfile
from PIL import Image
import time
from gradio_client import Client, file
import base64
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np
import uuid
import warnings

warnings.filterwarnings("ignore")

# Configuração da página
st.set_page_config(
    page_title="Barbearia Virtual - Face Swap",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tamanho padrão para as imagens
STANDARD_IMAGE_SIZE = (400, 400)  # Largura, Altura

# Estilo CSS personalizado
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #333;
    }
    .gallery-item {
        cursor: pointer;
        transition: transform 0.3s;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 5px;
    }
    .gallery-item:hover {
        transform: scale(1.05);
    }
    .result-container {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 10px 0;
    }
    .image-container img {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Inicialização de variáveis de sessão
if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None
if 'selected_style' not in st.session_state:
    st.session_state['selected_style'] = None
if 'result_image' not in st.session_state:
    st.session_state['result_image'] = None
if 'camera_image' not in st.session_state:
    st.session_state['camera_image'] = None
if 'processing' not in st.session_state:
    st.session_state['processing'] = False
if 'photo_mode' not in st.session_state:
    st.session_state['photo_mode'] = "Fazer upload de imagem"

# Função para redimensionar imagens mantendo a proporção
def resize_image(image, target_size):
    if image is None:
        return None
    
    if isinstance(image, np.ndarray):
        # Se for numpy array (da câmera), converter para PIL Image
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Redimensionar mantendo a proporção
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Criar uma nova imagem com fundo branco para o tamanho exato
    new_image = Image.new("RGB", target_size, (255, 255, 255))
    # Colar a imagem redimensionada no centro
    new_image.paste(
        image, 
        ((target_size[0] - image.width) // 2, (target_size[1] - image.height) // 2)
    )
    
    return new_image

# Função para salvar temporariamente uma imagem
def save_temp_image(image, format="JPEG"):
    if image is None:
        return None
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format.lower()}")
    image_path = temp_file.name
    
    if isinstance(image, np.ndarray):
        # Converter de numpy array para PIL Image
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    image.save(image_path, format=format)
    return image_path

# Função para realizar o face swap
def face_swap(source_path, target_path):
    try:
        # Cliente do modelo
        client = Client("felixrosberg/face-swap", hf_token=st.secrets["hungging"])
        
        # Executar o modelo
        result = client.predict(
            source=file(source_path),  # Sua foto (rosto)
            target=file(target_path),  # Estilo de cabelo desejado
            slider=100,  # Intensidade do swap
            adv_slider=100,  # Configurações avançadas
            settings=[],
            api_name="/run_inference"
        )
        
        # Carregar e retornar a imagem resultante
        result_img = Image.open(result)
        return result_img
    
    except Exception as e:
        st.error(f"Erro ao processar face swap: {str(e)}")
        return None

# Função para processar o quadro da webcam
class VideoProcessor:
    def __init__(self):
        self.current_frame = None
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Sempre manter o frame atual
        self.current_frame = img.copy()
        
        # Desenha um contorno para o rosto, para indicar o posicionamento
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Adicionar texto indicativo
                cv2.putText(img, "Rosto detectado", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        except:
            pass  # Ignorar erros de detecção de rosto
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Header
st.title("✂️ Barbearia Virtual - Face Swap")
st.subheader("Experimente novos cortes de cabelo virtualmente!")

# Criar colunas para layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📸 Sua Foto")
    
    # Opções para obter a imagem do usuário
    option = st.radio("Como deseja adicionar sua foto?", 
                      ["Fazer upload de imagem", "Usar câmera"],
                      key="photo_option")
    
    # Limpar imagem quando mudar de modo
    if option != st.session_state['photo_mode']:
        st.session_state['uploaded_image'] = None
        st.session_state['camera_image'] = None
        st.session_state['photo_mode'] = option
    
    if option == "Fazer upload de imagem":
        uploaded_file = st.file_uploader("Envie uma foto sua de frente", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Carregar a imagem e redimensionar
            user_image = Image.open(uploaded_file)
            user_image = resize_image(user_image, STANDARD_IMAGE_SIZE)
            
            # Salvar em session state
            st.session_state['uploaded_image'] = user_image
            st.session_state['camera_image'] = None  # Limpar imagem da câmera
            
            # Exibir a imagem
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(user_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:  # Usar câmera
        st.write("Posicione seu rosto no centro da câmera")
        
        # Container para o webrtc_streamer
        webrtc_ctx = webrtc_streamer(
            key="camera_snapshot",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Botões para controlar a câmera
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            # Botão para tirar foto
            if st.button("📸 Tirar Foto", key="take_photo"):
                if webrtc_ctx.video_processor and webrtc_ctx.video_processor.current_frame is not None:
                    # Capturar o frame atual
                    current_frame = webrtc_ctx.video_processor.current_frame.copy()
                    # Redimensionar e converter para PIL Image
                    snap_pil = resize_image(current_frame, STANDARD_IMAGE_SIZE)
                    st.session_state['camera_image'] = snap_pil
                    st.session_state['uploaded_image'] = None  # Limpar upload
                    st.success("Foto capturada com sucesso!")
                    st.rerun()
                else:
                    st.warning("Câmera não está ativa ou não há frame disponível. Aguarde um momento e tente novamente.")
        
        with col_btn2:
            # Botão para limpar foto
            if st.button("🗑️ Limpar Foto", key="clear_photo"):
                st.session_state['camera_image'] = None
                st.success("Foto removida!")
                st.rerun()
        
        # Exibir a foto capturada se existir
        if st.session_state['camera_image'] is not None:
            st.markdown("**Foto capturada:**")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(st.session_state['camera_image'], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Mostrar status da câmera
        if webrtc_ctx.state.playing:
            st.success("📹 Câmera ativa - Pronto para tirar foto")
        else:
            st.info("📷 Clique em 'START' para iniciar a câmera")

with col2:
    st.markdown("### 💇 Estilos de Corte Disponíveis")
    st.write("Navegue pelos estilos disponíveis:")

    # Diretório onde as imagens dos estilos estão armazenadas
    STYLES_DIR = "styles"  # Altere para o caminho do seu diretório

    # Dicionário de estilos organizados por categoria
    style_categories = {
        "Clássicos": {
            "Corte Side Part": "1.png",
            "Americano 1": "18.png",
            "Americano 2": "38.png",
            "Buzzcut 1": "14.png",
            "Buzzcut 2": "34.png",
            "Corte Militar 1": "3.png",
            "Classic Taper 1": "16.png",
            "Classic Taper 2": "36.png",
            "Caesar": "21.png",
            "Under Slicked-Back": "25.png",
            "Corte Militar 2": "6.png"
        },
        "Modernos": {
            "Corte Undercut 1": "2.png",
            "Corte Undercut 2": "10.png",
            "Corte Undercut 3": "30.png",
            "Old Money 1": "17.png",
            "Old Money 2": "37.png",
            "Pompadour 1": "11.png",
            "Pompadour 2": "31.png",
            "Surfista 1": "12.png",
            "Repicado": "22.png",
            "Surfista 2": "32.png",
            "Topete 1": "24.png",
            "Razor Part 1": "13.png",
            "Corte com Risco": "41.png",
            "Razor Part 2": "33.png",
            "Taper 1": "20.png",
            "Taper 2": "40.png",
            "Corte Degradê 1": "4.png",
            "Mid fade": "26.png",
            "Low fade 1": "27.png",
            "Low fade 2": "42.png",
            "Corte Degradê 2": "7.png"
        },
        "Longos": {
            "Topete 2": "8.png",
            "Mullet 1": "15.png",
            "Mullet 2": "35.png",
            "Topete 3": "9.png"
        },
        "Ousados": {
            "Quadrado ou Flat Top": "23.png",
            "Corte Raspado": "5.png",
            "Corte do Jaca": "28.png",
            "Fluffy Edgar 1": "29.png",
            "Fluffy Edgar 2": "43.png",
            "V 1": "39.png",
            "V 2": "19.png"
        }
    }

    # Criar abas para cada categoria
    tabs = st.tabs(list(style_categories.keys()))

    for tab, (category_name, styles) in zip(tabs, style_categories.items()):
        with tab:
            # Inicializar índice da página atual no session_state para cada categoria
            if f'style_page_{category_name}' not in st.session_state:
                st.session_state[f'style_page_{category_name}'] = 0

            # Número de estilos por página (3 colunas x 2 linhas = 6)
            STYLES_PER_PAGE = 6

            # Verificar se o diretório existe
            if not os.path.exists(STYLES_DIR):
                os.makedirs(STYLES_DIR)
                st.warning(f"Diretório '{STYLES_DIR}' criado. Por favor, adicione as imagens dos estilos.")
            else:
                # Lista de estilos para navegação
                style_list = list(styles.items())
                total_styles = len(style_list)
                total_pages = (total_styles + STYLES_PER_PAGE - 1) // STYLES_PER_PAGE

                # Botões de navegação
                if total_pages > 1:
                    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 3])
                    with col_nav1:
                        if st.button("⬅️ Anterior", key=f"prev_{category_name}", 
                                   disabled=st.session_state[f'style_page_{category_name}'] == 0):
                            st.session_state[f'style_page_{category_name}'] = max(0, st.session_state[f'style_page_{category_name}'] - 1)
                            st.rerun()
                    with col_nav2:
                        if st.button("Próximo ➡️", key=f"next_{category_name}", 
                                   disabled=st.session_state[f'style_page_{category_name}'] >= total_pages - 1):
                            st.session_state[f'style_page_{category_name}'] = min(total_pages - 1, st.session_state[f'style_page_{category_name}'] + 1)
                            st.rerun()
                    with col_nav3:
                        st.write(f"Página {st.session_state[f'style_page_{category_name}'] + 1} de {total_pages}")

                # Calcular os índices dos estilos a serem exibidos
                start_idx = st.session_state[f'style_page_{category_name}'] * STYLES_PER_PAGE
                end_idx = min(start_idx + STYLES_PER_PAGE, total_styles)

                # Exibir estilos em grid 3x2
                for row in range(2):  # 2 linhas
                    cols = st.columns(3)  # 3 colunas
                    for col_idx in range(3):  # Para cada coluna
                        style_idx = start_idx + (row * 3) + col_idx
                        if style_idx < end_idx:
                            style_name, style_file = style_list[style_idx]
                            style_path = os.path.join(STYLES_DIR, style_file)

                            # Verificar se o arquivo existe
                            if os.path.exists(style_path):
                                with cols[col_idx]:
                                    img = Image.open(style_path)
                                    # Redimensionar o estilo para exibição na galeria
                                    img = resize_image(img, STANDARD_IMAGE_SIZE)

                                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                                    st.image(img, caption=style_name, use_container_width=True)
                                    st.markdown('</div>', unsafe_allow_html=True)

                                    # Botão para selecionar o estilo
                                    if st.button(f"Selecionar {style_name}", key=f"style_{category_name}_{style_idx}"):
                                        # Salvar temporariamente a imagem selecionada
                                        temp_style_path = save_temp_image(img)
                                        st.session_state['selected_style'] = {
                                            'name': style_name,
                                            'path': temp_style_path,
                                            'image': img,
                                            'category': category_name
                                        }
                                        st.success(f"Estilo '{style_name}' selecionado!")
                                        st.rerun()

# Seção para processar e exibir o resultado
st.markdown("---")
st.markdown("### 🪄 Visualização do Resultado")

# Verificar se o usuário selecionou uma foto e um estilo
user_image = st.session_state.get('uploaded_image') or st.session_state.get('camera_image')
source_path = None

if user_image is not None and st.session_state.get('selected_style') is not None:
    # Preparar a imagem do usuário
    source_path = save_temp_image(user_image)
    
    # Exibir imagens lado a lado
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sua foto:**")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(user_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"**Estilo selecionado: {st.session_state['selected_style']['name']}**")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(st.session_state['selected_style']['image'], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Botão para processar o face swap
    if st.button("✨ Aplicar Face Swap") and not st.session_state['processing']:
        st.session_state['processing'] = True
        
        with st.spinner("Processando o face swap... Isso pode levar alguns segundos."):
            try:
                result_img = face_swap(
                    source_path=source_path,
                    target_path=st.session_state['selected_style']['path']
                )
                
                if result_img:
                    # Redimensionar o resultado
                    result_img = resize_image(result_img, STANDARD_IMAGE_SIZE)
                    st.session_state['result_image'] = result_img
                    
                    # Salvar o resultado em um arquivo temporário para download
                    result_path = save_temp_image(result_img)
                    
                    # Exibir o resultado
                    st.success("Face swap concluído com sucesso!")
                    
                    # Converter para base64 para exibição
                    buffered = BytesIO()
                    result_img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Exibir o resultado com HTML para melhor apresentação
                    st.markdown(f"""
                    <div class="result-container">
                        <h3 style="text-align: center;">🎉 Seu Novo Visual!</h3>
                        <div class="image-container">
                            <img src="data:image/jpeg;base64,{img_str}">
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Botão para download
                    with open(result_path, "rb") as file:
                        btn = st.download_button(
                            label="⬇️ Baixar Imagem",
                            data=file,
                            file_name=f"novo_visual_{uuid.uuid4().hex[:8]}.png",
                            mime="image/jpeg"
                        )
            
            except Exception as e:
                st.error(f"Ocorreu um erro durante o processamento: {str(e)}")
        
        st.session_state['processing'] = False

else:
    if user_image is None:
        st.info("👆 Primeiro, adicione sua foto usando uma das opções acima.")
    elif st.session_state.get('selected_style') is None:
        st.info("👆 Agora, selecione um estilo de corte da galeria acima.")

# Exibir resultado anterior se existir
if st.session_state.get('result_image') is not None:
    st.markdown("### 🎨 Último Resultado")
    st.image(st.session_state['result_image'], use_container_width=True)

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>✂️ Barbearia Virtual - Experimente antes de cortar! ✂️</p>
    <p style="font-size: 0.8em;">Esta aplicação usa tecnologia de face swap para visualização de cortes.</p>
    <p style="font-size: 0.8em;">Desenvolvido por Yagami Tecnologia - Whatsapp: 11-990000425</p>
</div>
""", unsafe_allow_html=True)

# Limpar arquivos temporários ao final
def cleanup():
    if st.session_state.get('uploaded_image'):
        if hasattr(st.session_state['uploaded_image'], 'name') and os.path.exists(st.session_state['uploaded_image'].name):
            os.unlink(st.session_state['uploaded_image'].name)
    
    if st.session_state.get('selected_style') and 'path' in st.session_state['selected_style']:
        if os.path.exists(st.session_state['selected_style']['path']):
            os.unlink(st.session_state['selected_style']['path'])

# Registrar função de limpeza para ser executada quando o app for fechado
import atexit
atexit.register(cleanup)