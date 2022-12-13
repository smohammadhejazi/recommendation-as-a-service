FROM python:3-alpine

RUN pip install flask numpy Cython kabirrec

WORKDIR /app

COPY /api_server .

CMD [ "python", "./server.py" ]
