# Get image
# Ensure to use 64-bit architecture otherwise cannot pip install catboost
FROM --platform=linux/amd64 python:3.9-slim

# Install dependencies
ADD requirements.txt /requirements.txt
RUN pip install -U pip && \
    pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install -r requirements.txt

# Copy files
COPY . .
