# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependency needed to unzip the dataset
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Extract the dataset so the app can read it at startup
RUN unzip -o data/archive.zip -d data/

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME=ProductSimilarityApp

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
