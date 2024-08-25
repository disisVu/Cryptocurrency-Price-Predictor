# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements.txt from the build directory to the /app directory in the container
COPY build/requirements.txt /usr/src/app/

# Install the required Python packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source folder contents into the /app directory in the container
COPY source/ /usr/src/app/

# Expose port for Flask
EXPOSE 5000

# Run main.py when the container launches
CMD ["python", "main.py"]
