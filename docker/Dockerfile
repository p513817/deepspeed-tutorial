FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

COPY requirements.txt /tmp/requirements.txt

RUN apt-get update && \
    apt-get install -yq git vim htop

RUN pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY inno-entrypoint /usr/bin/inno-entrypoint
RUN chmod u+x /usr/bin/inno-entrypoint

RUN adduser inno
USER inno
WORKDIR /inno

ENTRYPOINT [ "/usr/bin/inno-entrypoint" ]
CMD [ "bash" ]