from projectmonai/monai:latest

RUN pip install -U pip && \
    pip install pytorch-lightning>=1.5.0

COPY . /opt/manafaln
RUN cd /opt/manafaln && \
    python setup.py install && \
    rm -rf /workspace/* && \
    cp -r /opt/manafaln/examples /workspace/examples
