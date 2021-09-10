# MLOps
Parameter tunning:

Command to execute from root of repo: python mlops/hand-written-digits.py

Train size: 1527, Test size: 270, Val size: 270

## Gamma :0.0001
Result on validation set
Accuracy : 0.9111111111111111, F1 Score :0.909137570548868

## Gamma :0.001
Result on validation set
Accuracy : 0.9481481481481482, F1 Score :0.9475460219925551

## Gamma :0.01
Result on validation set
Accuracy : 0.6666666666666666, F1 Score :0.7210949727848763

## Gamma :0.1
Result on validation set
Accuracy : 0.0962962962962963, F1 Score :0.01756756756756757

## Gamma :1.0
Result on validation set
Accuracy : 0.0962962962962963, F1 Score :0.01756756756756757


## Best Gamma is : 0.001 with accuracy of 0.9481481481481482 and F1 score of 0.9475460219925551

Result on Test set
Accuracy : 0.9481481481481482, F1 Score :0.9475460219925551

Result on Train set
Accuracy : 0.9993451211525868, F1 Score :0.9993442340976768