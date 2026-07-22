# Use a lightweight Python image
FROM python:3.10-slim

# Install system libraries required by OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Set the working directory
WORKDIR /app

# Copy your files to the cloud
COPY . /app

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Run the Flask app using Gunicorn (production server)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 server:app