import streamlit as st
from streamlit_image_comparison import image_comparison
import numpy as np
st.set_page_config(page_title='pont.tech demo')

import cv2
import os
from uuid import uuid4
import torch
import shutil

from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

print(torch.cuda.is_available())

from rife.RIFE_HDv3 import Model
RIFE = Model()
RIFE.load_model("./rife/", -1)
print("Loaded v3.x HD model.")

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

@st.cache_resource
def import_ESRGAN():
    from realesrgan import RealESRGANer
    return RealESRGANer


if st.session_state.get("uuid") is None:
    st.session_state["uuid"] = uuid4()

path = f'./media/{st.session_state["uuid"]}'
os.makedirs(path, exist_ok=True)

import subprocess

def convert_to_mp4(image_folder, output_file, filename="result_%d.png", fps=30):
    subprocess.run(['ffmpeg', '-i', f'{image_folder}/{filename}', '-y', '-framerate', f'{fps}', output_file])

RealESRGANer = import_ESRGAN()

from stqdm import stqdm

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
netscale = 2
file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'

ROOT_DIR = os.path.dirname(os.path.curdir)
model_path = load_file_from_url(url=file_url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

upsampler = RealESRGANer(
    scale=2,
    model_path=model_path,
    model=model,
    gpu_id=0)

def list_files_with_full_path(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = f"{path}/{file}"
            file_list.append(file_path)
    return file_list

def clear_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)

def get_color_brightness(hex_code):
    # Remove the '#' symbol if present
    hex_code = hex_code.lstrip('#')
    
    # Convert hexadecimal to RGB values
    r, g, b = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    
    # Calculate the brightness using the formula
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    
    return brightness

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


images = sorted(filter(lambda f: ".png" in f and "input" not in f, list_files_with_full_path(path)))
exp = 1
N = 2 ** exp
bc = st.get_option('theme.backgroundColor')
bc_brightness = get_color_brightness(bc) if bc is not None else 1
st.image("pont.tech_logo.png" if bc_brightness > 50 else "pont.tech_logo_white.png", width=200)
if check_password():
    st.markdown("This application is designed to enhance the quality of frames in a sequence through upscaling and interpolation technology, which is utilized by pont.tech cloud. To get started, you need to upload a sequence of files by selecting the **\"Upload\"** button.")
    st.markdown("In case you don't have your own sequence, there are some prepared sample files available for you to use.")
    st.markdown("[Car sample](https://drive.google.com/file/d/1y8DNc9Smo8cx2kFxtQ_7DL8RJ7qxKvzm/view?usp=drive_link)        [Plant sample](https://drive.google.com/file/d/1cuLIal-bquihelrvT3KfCp623WgUoTke/view?usp=drive_link)")
    st.markdown("It is important to note that the current version of the app only supports X2 upscaling, but the interpolation capabilities are unlimited.")
    with st.form("my-form", clear_on_submit=True):
        files = st.file_uploader("FILE UPLOADER", accept_multiple_files=True)
        submitted = st.form_submit_button("UPLOAD!")
        insert_frames = st.number_input("Interpolation scale", min_value=1, max_value=4, value=2, disabled=submitted)
        base_fps = st.number_input("Base sequence FPS", min_value=6, max_value=60, value=15, disabled=submitted)

        exp = insert_frames - 1
        N = 2 ** exp
        image_placeholder = st.empty()

        if submitted and files is not None:
            clear_directory(path)
            with st.spinner('Wait for it...'):
                images = []
                img_prev = None
                img_np = None
                img_result = None
                for i, uploaded_file in enumerate(stqdm(sorted(files, key=lambda f: f.name), desc="Processing...")):
                    k_prev = (i-1) * N
                    k = i * N
                    bytes_data = uploaded_file.read()
                    nparr = np.fromstring(bytes_data, np.uint8)
                    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    img_result, _ = upsampler.enhance(img_np)
                    if img_prev is not None:

                        for j in range(N-1):
                            img0 = (torch.tensor(img_prev.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
                            img1 = (torch.tensor(img_result.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
                            n, c, h, w = img0.shape
                            ph = ((h - 1) // 64 + 1) * 64
                            pw = ((w - 1) // 64 + 1) * 64
                            padding = (0, pw - w, 0, ph - h)
                            img0 = F.pad(img0, padding)
                            img1 = F.pad(img1, padding)
                            img_intrp = RIFE.inference(img0, img1, (j+1) * 1. / N)
                            cv2.imwrite(f"{path}/result_{k_prev + (j + 1)}.png", (img_intrp[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
                            images.append(f"{path}/result_{k_prev + (j + 1)}.png")
                    img_prev = img_result
                    cv2.imwrite(f"{path}/result_{k}.png", img_result)
                    cv2.imwrite(f"{path}/input_{i}.png", img_np)
                    images.append(f"{path}/result_{k}.png")
                    image_placeholder.image(images[-6:], width=100)
            image_placeholder.image([])
            if len(files) > 1:
                with st.spinner('Making MP4...'):
                    convert_to_mp4(path, f'{path}/output.mp4', filename="result_%d.png", fps=base_fps * N)
                    convert_to_mp4(path, f'{path}/input.mp4', filename="input_%d.png", fps=base_fps)
            
            with st.spinner('Making Archive...'):
                shutil.make_archive(path, 'zip', path)
        submitted = False

    images = sorted(filter(lambda f: ".png" in f and "input" not in f, list_files_with_full_path(path)))
    num_frames = len(images)
    if num_frames > 0:
        tab1, tab2 = st.tabs(["Inspect", "Download"])
        with tab1:
            tab11, tab12, tab13, tab14 = st.tabs(["Upscaling", "Interpolation", "Video", "Frames"])
            with tab11:
                with st.form(key="Streamlit Image Comparison"):
                    if num_frames > 1:
                        frame = st.select_slider("Frame", options=list(range(0, num_frames, N)), value=0)
                    else:
                        frame = 0
                    submit = st.form_submit_button("Update Render 🔥")
                    comp = image_comparison(
                        img1=f"{path}/input_{frame // N}.png", 
                        img2=f"{path}/result_{frame}.png", width=600,
                        label1="Original",
                        label2="Upscaled"
                        )
            with tab12:
                if num_frames > 1:
                    frame = st.select_slider("Frame", options=list(range(0, num_frames)), value=0)
                else:
                    frame = 0
                st.image(f"{path}/result_{frame}.png", width=600)

            with tab13:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Input")
                    if os.path.exists(f'{path}/input.mp4'):
                        st.video(f'{path}/input.mp4')
                with col2:
                    st.write("Result")
                    if os.path.exists(f'{path}/output.mp4'):
                        st.video(f'{path}/output.mp4')


            with tab14:
                st.image(sorted(images), width=100)

        with tab2:
            col1, col2 = st.columns([1,1])
            with col1:
                if os.path.exists(f'{path}/output.mp4'):
                    with open(f'{path}/output.mp4', "rb") as file:
                        download_mp4 = st.download_button(
                                label="Download MP4",
                                data=file,
                                file_name='result.mp4',
                                mime="video/mp4"
                        )
            with col2:
                if os.path.exists(f'{path}.zip'):
                    with open(f'{path}.zip', "rb") as file:
                        download_mp4 = st.download_button(
                                label="Download ZIP",
                                data=file,
                                file_name='result.zip',
                                mime="application/zip"
                        )

