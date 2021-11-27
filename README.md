# MLOps
## Docker predict SVM and DT

### Image build command 
sudo docker build -t ass10 . -f docker/Dockerfile

### Container RUN Command
sudo docker run -d --network=host ass10:latest