# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container to /app
WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Copy the current directory contents into the container at /app
COPY stick-it-ai-server.py /app/stick-it-ai-server.py

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make sure the model directory exists
RUN mkdir -p /app/model

VOLUME [ "/app/model" ]

# Expose the port that FastAPI will listen on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["python", "-m", "uvicorn", "stick-it-ai-server:app", "--host", "0.0.0.0", "--port", "8000"]
