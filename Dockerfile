# Use an official Python image as the base
# Alterei a versão que veio da profa para bater com meu ambiente
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn
RUN apt-get update && apt-get -y install ffmpeg libsm6 libxext6

# Copy the application code
# Além da aplicação que fiz, também adicionei o .pkl com o modelo Pickle que treinei no notebook
COPY . .


# Expose the port
EXPOSE 8000

# Run the command to start the development server
CMD [ "gunicorn", "-b", "0.0.0.0:8000", "main:app" ]