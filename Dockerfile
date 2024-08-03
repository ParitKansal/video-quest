FROM python:3.11-slim

WORKDIR /app

COPY ./ /app

RUN pip install --no-cache-dir -r requirements.txt

COPY cipher.py /usr/local/lib/python3.11/site-packages/pytube/cipher.py

CMD ["streamlit", "run", "application-summary.py", "--server.port=80", "--server.address=0.0.0.0"]