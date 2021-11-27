FROM ubuntu:20.04

RUN mkdir /code

WORKDIR /code

copy ./mlops mlops/

copy ./requirements.txt .

RUN mkdir model

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && apt-get clean

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "mlops/digits.py"]