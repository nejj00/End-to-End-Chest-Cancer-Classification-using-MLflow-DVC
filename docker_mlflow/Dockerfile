FROM python:3.10-slim-buster

RUN apt update -y && apt install awscli -y

# Set the AIRFLOW_HOME environment variable to the root directory
ENV AIRFLOW_HOME=/app

WORKDIR /app

# Install app dependencies
COPY docker_mlflow/requirements.txt .

RUN pip install -r requirements.txt

# Bundle app source
COPY . .

EXPOSE 5000

CMD [ "mlflow", "server","--host","0.0.0.0","--port","5000"]