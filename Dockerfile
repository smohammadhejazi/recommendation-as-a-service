FROM python:3.10-slim-bullseye as base

WORKDIR /app

# ---- Dependencies ----
FROM base AS dependencies  
COPY ./requirements.txt ./
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir flask

# ---- Build ----
FROM dependencies as build
WORKDIR /app

RUN apt-get update
RUN apt-get install -y gcc
COPY ./requirements.txt ./setup.py ./
COPY ./kabirrec ./kabirrec
RUN pip install --user .

# --- Release ----
FROM dependencies as release
WORKDIR /app

COPY --from=build /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY ./api_server ./

EXPOSE 8080
CMD [ "python", "server.py" ]
