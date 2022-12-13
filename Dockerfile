FROM python:3-alpine

RUN pip install flask kabirrec

WORKDIR /app

COPY /api_server .

CMD [ "python", "./server.py" ]
