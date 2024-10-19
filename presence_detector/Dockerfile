FROM python:3.9-slim-buster

RUN pip3 install --no-cache-dir pandas scikit-learn joblib requests

COPY run.sh /
COPY ml_presence_detector.py /

RUN chmod a+x /run.sh

CMD [ "/run.sh" ]
