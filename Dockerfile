# syntax=docker/dockerfile:1

FROM python:3.11

WORKDIR /python-docker

COPY ./templates /python-docker
COPY ./image_classifier.keras /python-docker
COPY ./static /python-docker

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install ca-certificates
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
EXPOSE 5000
ENV FLASK_APP=api.py
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
