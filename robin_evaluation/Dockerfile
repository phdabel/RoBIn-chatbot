FROM python:3.11-slim

WORKDIR /app

COPY ./src/ /app

RUN mkdir -p /data/evaluation
RUN mkdir -p /data/pmc

COPY ./pyproject.toml /code/pyproject.toml
RUN pip install /code/.

CMD ["sh", "entrypoint.sh"]