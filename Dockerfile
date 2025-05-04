FROM python:3.11-slim

COPY ./requirements.txt /webapp/requirements.txt

WORKDIR /medgan

RUN pip install -r requirements.txt

COPY * /medgan

