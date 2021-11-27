# MLOps
## EXAM

### Model Hyperparameter command

python mlops/digits.py

#### The best Gama is : 0.001 and 0.01 is overfitting that's why also perform bad on Test and validation

### Image build command 
sudo docker build -t exam . -f docker/Dockerfile

### Container RUN Command
sudo docker run -d --network=host --name exam -v '/media/sachin/DATA1/IITJ/Sem IV/MLOps/Task/MLOps/modelExam':/code/modelExam exam:latest