# Use an official Python runtime as a parent image
FROM python:3.10

# Copy the rest of the application files
COPY . /app

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# This command creates a .streamlit directory in the home directory of the container.
RUN mkdir ~/.streamlit

# This copies your Streamlit configuration file into the .streamlit directory you just created.
RUN cp config.toml ~/.streamlit/config.toml

# Similar to the previous step, this copies your Streamlit credentials file into the .streamlit directory.
RUN cp credentials.toml ~/.streamlit/credentials.toml

# Run streamlit when the container launches
CMD ["streamlit", "run", "streamlit_app.py"]