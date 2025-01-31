#!/bin/bash

# Install ComfyUI-Depth-Anything-Tensorrt Node
cd /comfyui/custom_nodes
git clone https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt.git
cd ComfyUI-Depth-Anything-Tensorrt
pip install -r requirements.txt

# Install ComfyUI_RyanOnTheInside
cd /comfyui/custom_nodes
git clone https://github.com/pschroedl/ComfyUI_RyanOnTheInside.git
cd ComfyUI_RyanOnTheInside
git checkout c5592719d5188cb6bba81287a8af123637eadf79
pip install -r requirements.txt

# Install ComfyUI-Misc-Effects (Starburst)
cd /comfyui/custom_nodes
git clone https://github.com/ryanontheinside/ComfyUI-Misc-Effects.git
cd ComfyUI-Misc-Effects
git checkout c6b360c78611134c3723388170475eb4898ff6b7

# Install necessary packages
pip install torch==2.5.1 torchvision torchaudio tqdm nvidia-ml-py==12.560.30

# Install comfystream (which includes ComfyUI)
pip install git+https://github.com/yondonfu/comfystream.git
cd /comfyui/custom_nodes
git clone https://github.com/yondonfu/comfystream.git
cd comfystream
pip install -r requirements.txt
cp -r nodes/tensor_utils /comfyui/custom_nodes/

# Install ComfyUI-SAM2-Realtime
cd /comfyui/custom_nodes
git clone https://github.com/pschroedl/ComfyUI-SAM2-Realtime.git
cd ComfyUI-SAM2-Realtime
git checkout 4f587443fb2808c4b5b303afcd7ec3ec3e0fbd08
pip install -r requirements.txt

# Install ComfyUI-Florence2-Vision
cd /comfyui/custom_nodes
git clone https://github.com/ad-astra-video/ComfyUI-Florence2-Vision.git
cd ComfyUI-Florence2-Vision
git checkout 0c624e61b6606801751bd41d93a09abe9844bea7
pip install -r requirements.txt

# Install ComfyUI-StreamDiffusion
cd /comfyui/custom_nodes
git clone https://github.com/pschroedl/ComfyUI-StreamDiffusion.git
cd ComfyUI-StreamDiffusion
git checkout f93b98aa9f20ab46c23d149ad208d497cd496579
pip install -r requirements.txt

# Install ComfyUI-LivePortraitKJ Node
pip install diffusers==0.30.1
cd /comfyui/custom_nodes
git clone https://github.com/kijai/ComfyUI-LivePortraitKJ.git
cd ComfyUI-LivePortraitKJ
git checkout 4d9dc6205b793ffd0fb319816136d9b8c0dbfdff
pip install -r requirements.txt

# Install ComfyUI-load-image-from-url Node
cd /comfyui/custom_nodes
git clone https://github.com/tsogzark/ComfyUI-load-image-from-url.git
