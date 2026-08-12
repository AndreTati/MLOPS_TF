FROM apache/airflow:3.0.2
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip "setuptools<81" wheel
RUN pip install --no-cache-dir apache-airflow[virtualenv]==3.0.2 -r /requirements.txt