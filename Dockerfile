FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY . /app

RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_APP=app.py
ENV FLASK_ENV=chatbot

RUN python3 setup.py develop

EXPOSE 5003

CMD ["python3","app.py"]

