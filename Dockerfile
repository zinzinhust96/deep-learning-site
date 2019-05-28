FROM python:3.6.7-alpine

WORKDIR /opt/app

RUN mkdir -p /opt/app && cd /opt/app

COPY requirements.txt /
# Install Requirements
RUN pip install --no-cache -r requirements.txt

COPY uwsgi.ini /etc/uwsgi/

# Copy the code
COPY . .

CMD ['uwsgi', '--ini', '/etc/uwsgi/uwsgi.ini']

EXPOSE 5000