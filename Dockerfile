FROM python:3.10.4-alpine3.15

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY ./requirements.txt ./

RUN pip install --no-cache-dir -r ./requirements.txt
RUN apk upgrade \
    && apk add build-base \
    && apk add build-base cmake git gtk+2.0-dev pkgconfig ffmpeg-dev

RUN python3 -m pip install --force-reinstall --no-cache opencv-python==4.7.0.72 

RUN apk add --upgrade tesseract-ocr 
    
COPY . ./

CMD [ "python", "manage.py", "runserver", "0.0.0.0:3000" ]