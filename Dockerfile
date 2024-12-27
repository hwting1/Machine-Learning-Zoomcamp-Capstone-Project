FROM python:3.11.10-slim

WORKDIR /project

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "predict.py"]