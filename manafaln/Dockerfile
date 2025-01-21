from projectmonai/monai

COPY . /opt/manafaln
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir /opt/manafaln && \
    rm -rf /workspace/* && \
    cp -r /opt/manafaln/examples /workspace/examples
