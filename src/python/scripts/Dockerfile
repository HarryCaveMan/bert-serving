ARG BASE=trt-test

FROM $BASE

COPY requirements.txt requirements.txt

RUN dnf -y install python3.11-pip && ln /usr/bin/pip3.11 /usr/bin/pip &&\
    pip install --no-deps -r requirements.txt

COPY *.py /opt/service/