FROM continuumio/miniconda3:4.6.14

# Install Requirements
COPY requirements.txt .
RUN conda install -y -c conda-forge --quiet --file requirements.txt

WORKDIR /opt/app

RUN mkdir -p /opt/app && cd /opt/app

COPY uwsgi.ini /etc/uwsgi/

# Copy the code
COPY . .

CMD ["uwsgi", "--ini", "/etc/uwsgi/uwsgi.ini"]

EXPOSE 5000