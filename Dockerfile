# set base image (host OS)
FROM python:3.8.0-slim

# install some packages, for potential data fetching
RUN  apt-get update && apt-get install -y wget && apt-get install -y unzip

RUN useradd -m karasu

USER karasu

# set the working directory in the container
WORKDIR /home/karasu/app

ENV PATH="/home/karasu/.local/bin:${PATH}"
ENV PYTHONPATH "${PYTHONPATH}:/home/karasu/app"

# install dependencies
COPY --chown=karasu:karasu requirements.txt .
RUN pip install --user -r requirements.txt

# Add everything else now
COPY --chown=karasu:karasu . .

# command to run on container start
CMD ping localhost