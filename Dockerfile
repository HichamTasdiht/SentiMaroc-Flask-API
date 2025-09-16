# Use a slim Python 3.10 image as the base.
FROM python:3.10-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file and install all dependencies.
# This must be done before copying the rest of the code.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the project files into the container.
COPY . .

# Expose the port the Flask app will run on.
EXPOSE 5000

# Define the command to start the application using Gunicorn.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
