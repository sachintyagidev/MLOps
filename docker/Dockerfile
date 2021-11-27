FROM ubuntu:18.04

copy mlops /mlops/

copy api /api/

copy requirements.txt /mlops/

copy model/model_svm_0.0001.sav /mlops/model/

copy model/model_DT_14.sav /mlops/model/

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && apt-get clean

WORKDIR /mlops

RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /api

CMD ["flask", "run"]