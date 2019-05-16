FROM python:3.6.4

ADD requirements.txt /requirements.txt
ADD inferencer/* /inferencer/
ADD outputs /outputs/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000


CMD python inferencer/app.py


