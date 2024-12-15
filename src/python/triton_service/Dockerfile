FROM nvcr.io/nvidia/tritonserver:24.11-py3

COPY requirements.txt  .

RUN pip install --no-cache -r requirements.txt

COPY *.py /opt/service/

WORKDIR /opt/service
