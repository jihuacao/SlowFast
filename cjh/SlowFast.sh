# base on git@github.com:facebookresearch/SlowFast.git 2efb99faa254075b4e28d3d4f313052b51da05bc
# base payforsins/nvidia-dev:nvidia_dev_cuda118-dev-1.0.1-e0d8f9f-uncommited
apt update
apt -y install libopencv-dev
pip3 install --proxy 'socks5://172.29.129.38:1080' torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install --proxy 'socks5://172.29.129.38:1080' fvcore
pip3 install --proxy 'socks5://172.29.129.38:1080' numpy
pip3 install --proxy 'socks5://172.29.129.38:1080' simplejson
pip3 install --proxy 'socks5://172.29.129.38:1080' PyAV
pip3 install --proxy 'socks5://172.29.129.38:1080' iopath
pip3 install --proxy 'socks5://172.29.129.38:1080' psutil
pip3 install --proxy 'socks5://172.29.129.38:1080' opencv-python
pip3 install --proxy 'socks5://172.29.129.38:1080' moviepy
pip3 install --proxy 'socks5://172.29.129.38:1080' 'git+https://github.com/facebookresearch/fairscale'
pip3 install --proxy 'socks5://172.29.129.38:1080' detectron2_repo
pip3 install --proxy 'socks5://172.29.129.38:1080' onnx onnxruntime
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
cd detectron2_repo && git reset --hard e9f7e2ba15abd7badcb05ef6f5076f06b36a9c5b && git clean -df && cd ..
pip install -e detectron2_repo
git clone https://github.com/facebookresearch/pytorchvideo.git pytorchvideo
cd pytorchvideo && git reset --hard eb04d1b21e08cfd0713164c0907aeb4c98fd83af && git clean -df && cd ..
pip install -e pytorchvideo