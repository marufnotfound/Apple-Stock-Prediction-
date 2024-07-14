# Use the official Python base image
FROM python:3.9

# # Set the working directory inside the container
# WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY api.py .

# Expose the port on which the Flask server will listen
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "api.py"]
