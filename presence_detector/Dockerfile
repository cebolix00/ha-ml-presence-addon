FROM python:3.9-slim-buster

ENV LANG C.UTF-8

# Install required packages
RUN pip3 install --no-cache-dir pandas scikit-learn xgboost lightgbm catboost joblib requests

# Copy data for add-on
COPY run.sh presence_detector.py /

RUN chmod a+x /run.sh

CMD [ "/run.sh" ]
