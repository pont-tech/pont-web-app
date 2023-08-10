# Use an official Python runtime as the parent image
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && \
    apt-get install -y git libsm6 libxext6 libgl1 libx264-dev libfdk-aac-dev libass-dev libopus-dev libtheora-dev libvorbis-dev libvpx-dev libssl-dev ffmpeg && \
    git clone https://github.com/xinntao/Real-ESRGAN.git && cd Real-ESRGAN && python setup.py develop

# Copy the rest of the application code to the container
COPY . .

# Expose the port on which the Streamlit app will run (default is 8501)
EXPOSE 8501

# Set the entrypoint command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
