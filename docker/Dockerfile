FROM ubuntu:20.04

RUN mkdir /code

WORKDIR /code

copy ./mlops mlops/

copy ./api api/

copy ./requirements.txt mlops/

RUN mkdir model

copy ./model/model_svm_0.0001.sav model/

copy ./model/model_DT_14.sav model/

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && apt-get clean

WORKDIR /code/mlops

RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /code/api

EXPOSE 5000/tcp
EXPOSE 5000/udp

CMD ["flask", "run"]