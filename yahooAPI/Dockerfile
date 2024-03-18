FROM python:3.9.18-slim-bookworm
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "main.py"]