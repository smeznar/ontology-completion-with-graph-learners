# syntax=docker/dockerfile:1

FROM python:3.6

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch==1.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html


ENTRYPOINT ["python"]
