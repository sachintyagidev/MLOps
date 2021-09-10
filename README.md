# MLOps
Parameter tunning:

Command to execute from root of repo: python mlops/hand-written-digits.py

Train size: 1258, Test size: 270, Val size: 269

## Gamma :0.0001
Result on validation set
Accuracy : 0.9553903345724907, F1 Score :0.9545831499566034

## Gamma :0.001
Result on validation set
Accuracy : 0.9925650557620818, F1 Score :0.9926484660849987

## Gamma :0.01
Result on validation set
Accuracy : 0.758364312267658, F1 Score :0.7975424047634962

## Gamma :0.1
Result on validation set
Accuracy : 0.10037174721189591, F1 Score :0.018243243243243244

## Gamma :1.0
Result on validation set
Accuracy : 0.10037174721189591, F1 Score :0.018243243243243244


## Best Gamma is : 0.001 with accuracy of 0.9925650557620818 and F1 score of 0.9926484660849987

Result on Test set
Accuracy : 0.9481481481481482, F1 Score :0.9477475781671046

Result on Train set
Accuracy : 0.9992050874403816, F1 Score :0.9992031872509962
