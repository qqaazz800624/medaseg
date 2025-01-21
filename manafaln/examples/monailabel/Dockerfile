from projectmonai/monailabel:0.8.5

# For TensorRT acceleration
RUN pip install torch-tensorrt tensorrt \
    --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install nvidia-modelopt[torch]

# manafaln dependencies
RUN pip install "ruamel.yaml==0.18.6" "torchmetrics==1.4.0" "timm>=1.0.3" "lightning>=2.2.5" && \
    pip install --no-deps git+https://gitlab.com/nanaha1003/manafaln.git

# Remove pip installation cache to reduce image size
RUN pip cache purge

WORKDIR /workspace
