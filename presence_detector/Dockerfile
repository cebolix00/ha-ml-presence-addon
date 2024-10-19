FROM python:3.9-slim-buster

RUN pip3 install --no-cache-dir pandas scikit-learn joblib requests

COPY run.sh /
COPY ml_presence_detector.py /
COPY presence_detector/run.sh /etc/services.d/ml_presence_detector/run
RUN chmod a+x /run.sh

CMD [ "/run.sh" ]
