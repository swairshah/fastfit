FROM python:3.8-slim

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ARG model_id
ARG dataset

WORKDIR /app

COPY . .

ENTRYPOINT ["python", "train.py"]
#CMD ["model_name_or_path", "train_file", "validation_file"]
