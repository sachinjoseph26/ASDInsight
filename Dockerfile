# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the rest of the application files
COPY . /app

# Copy the .env file
COPY .env /app/.env

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gcc \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create .streamlit directory and copy configuration files
RUN mkdir ~/.streamlit
COPY streamlit_ui/config.toml ~/.streamlit/config.toml
COPY streamlit_ui/credentials.toml ~/.streamlit/credentials.toml

# Expose ports for Flask and Streamlit
EXPOSE 7811 7801 80

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set environment variables
ENV ENV_FILE_LOCATION=/app/.env

# Run the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
