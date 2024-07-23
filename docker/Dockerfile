FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

RUN apt-get update && \
    apt-get install -yq git vim htop numactl net-tools libaio-dev

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY inno-entrypoint /usr/local/bin/inno-entrypoint
RUN chmod +x /usr/local/bin/inno-entrypoint

RUN adduser --disabled-password --gecos "" --home /home/inno inno
RUN echo "/usr/local/bin/inno-entrypoint" >> /home/inno/.bashrc

USER inno
WORKDIR /opt/inno

# ENTRYPOINT [ "/usr/local/bin/inno-entrypoint" ]
CMD [ "bash" ]
