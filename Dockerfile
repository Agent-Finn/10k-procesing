# Use official Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app app/
COPY ./.env .env

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]




# # Use official Python image
# FROM python:3.11

# # Set the working directory inside the container
# WORKDIR /app

# # Copy the shared requirements.txt from the build context
# COPY requirements.txt /app/requirements.txt

# # Install dependencies
# RUN pip install --no-cache-dir -r /app/requirements.txt

# # Copy the microservice code into the container
# COPY . .

# # Expose the port FastAPI runs on
# EXPOSE 8000

# # Run the application
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]