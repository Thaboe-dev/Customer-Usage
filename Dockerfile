FROM python:3.9

WORKDIR /app

# copy the requirements
COPY ./requirements.txt /app/requirements.txt

# install dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# copy the code
COPY . /app/

# start command
CMD ["fastapi", "run", "/app/server/service.py", "--port", "80"]