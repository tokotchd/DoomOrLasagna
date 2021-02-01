FROM ubuntu:18.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update -y && \
    apt-get install -y python3 python3-pip ffmpeg libsm6 libxext6

COPY . /app
WORKDIR /app
run pip3 install --upgrade pip
run pip3 install --no-cache-dir -r requirements.txt

# use gunicorn instead of flask for more secure and higher performance hosting
CMD gunicorn -w 1 -b 0.0.0.0:80 app:app
# CMD python3 -m flask run --host=127.0.0.1
