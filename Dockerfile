FROM python:3.8

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
# RUN dvc init --no-scm
# RUN dvc pull -r myremote

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
