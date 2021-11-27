# MLOps
## EXAM

### Image build command 
sudo docker build -t exam . -f docker/Dockerfile

### Container RUN Command
sudo docker run -d --network=host --name exam -v '/media/sachin/DATA1/IITJ/Sem IV/MLOps/Task/MLOps/modelExam':/code/modelExam exam:latest

### Model Hyperparameter command

python mlops/digits.py

#### The best Gama is : 0.001 and 0.01 is overfitting that's why also perform bad on Test and validation

Hyperparameter : 0.0001

MPL Result : Train : 1.0 Test : 0.9555555555555556 Validation : 0.9666666666666667

Run 1, Train : 0.9777247414478918, Test : 0.9592592592592593, Validation : 0.9666666666666667

MPL Result : Train : 1.0 Test : 0.9740740740740741 Validation : 0.9777777777777777

Run 2, Train : 0.9761336515513126, Test : 0.9777777777777777, Validation : 0.9666666666666667

MPL Result : Train : 1.0 Test : 0.9703703703703703 Validation : 0.9629629629629629

Run 3, Train : 0.9785202863961814, Test : 0.9592592592592593, Validation : 0.9777777777777777

Mean :Train : 0.9774595597984619, Test : 0.9654320987654321, Validation : 0.9703703703703703

Hyperparameter : 0.001

MPL Result : Train : 1.0 Test : 0.9592592592592593 Validation : 0.9888888888888889

Run 1, Train : 0.9992044550517104, Test : 0.9888888888888889, Validation : 0.9962962962962963

MPL Result : Train : 1.0 Test : 0.9666666666666667 Validation : 0.9555555555555556

Run 2, Train : 0.9984089101034208, Test : 0.9814814814814815, Validation : 0.9740740740740741

MPL Result : Train : 1.0 Test : 0.9666666666666667 Validation : 0.9777777777777777

Run 3, Train : 1.0, Test : 0.9888888888888889, Validation : 0.9888888888888889

Mean :Train : 0.9992044550517104, Test : 0.9864197530864198, Validation : 0.9864197530864198

Hyperparameter : 0.01

MPL Result : Train : 1.0 Test : 0.9592592592592593 Validation : 0.9629629629629629

Run 1, Train : 1.0, Test : 0.6444444444444445, Validation : 0.7407407407407407

MPL Result : Train : 1.0 Test : 0.9740740740740741 Validation : 0.9888888888888889

Run 2, Train : 1.0, Test : 0.7814814814814814, Validation : 0.7148148148148148

MPL Result : Train : 1.0 Test : 0.9925925925925926 Validation : 0.9666666666666667

Run 3, Train : 1.0, Test : 0.8185185185185185, Validation : 0.8296296296296296

Mean :Train : 1.0, Test : 0.7481481481481481, Validation : 0.7617283950617284