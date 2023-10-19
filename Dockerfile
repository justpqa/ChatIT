# Use an official Jupyter Notebook base image
FROM python:3.8

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=development

# install c compiler for llama-cpp-python
USER root
RUN apt-get update && \
    apt-get -y install gcc cmake g++

# select work directory
WORKDIR /src

# Install additional Python packages from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy folder
COPY . /src

# expose port
EXPOSE 5000

# run the app
CMD ["flask", "run", "--host=0.0.0.0"]

# legacy code for jupyter env
# FROM jupyter/base-notebook:latest
# RUN echo "c.NotebookApp.token = ''" >> /home/jovyan/.jupyter/jupyter_notebook_config.py && \
#     echo "c.NotebookApp.password = ''" >> /home/jovyan/.jupyter/jupyter_notebook_config.py
# WORKDIR /home/jovyan/work
# USER root
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]