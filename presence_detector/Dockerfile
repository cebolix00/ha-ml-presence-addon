FROM python:3.9-slim-buster

RUN pip3 install --no-cache-dir pandas scikit-learn joblib requests numpy

COPY presence_detector.py /presence_detector.py
COPY run.sh /run.sh

RUN chmod a+x /run.sh

CMD [ "/run.sh" ]
