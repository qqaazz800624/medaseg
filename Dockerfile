from nvcr.io/nvidia/pytorch:21.10-py3

RUN pip install -U pip && \
    pip install monai[all]==0.6.0 && \
    pip install pytorch-lightning==1.4.9

COPY . /opt/manafaln
RUN cd /opt/manafaln && \
    python setup.py install && \
    rm -rf /workspace/* && \
    cp -r /opt/manafaln/examples /workspace/examples
