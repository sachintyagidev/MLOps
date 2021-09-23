# MLOps
## Modularization:

Command to execute from root of repo: 

python mlops/hand-written-digits.py --testSize 0.15 --valSize 0.15 --modelLoc 'model' --metricLoc 'model' --gamaSet 1e-07 1e-06 1e-05 0.0001 0.001 0.01 0.1

### Parameters Details

--testSize      Test Set Size

--valSize'      Validation Size

--modelLoc      Model Save Location

--metricLoc     Metric Save Location

--gamaSet       Gamma List to Test


### Train size: 1297, Test size: 270, Val size: 230

## Gamma :1e-07
Result on validation set

Accuracy : 0.10869565217391304, F1 Score :0.0196078431372549

## Gamma :1e-06
Result on validation set

WAccuracy : 0.10869565217391304, F1 Score :0.0196078431372549

## Gamma :1e-05
Result on validation set

Accuracy : 0.8782608695652174, F1 Score :0.8793811717364187

## Gamma :0.0001
Result on validation set

Accuracy : 0.9608695652173913, F1 Score :0.9605782468025972

## Gamma :0.001
Result on validation set

Accuracy : 0.991304347826087, F1 Score :0.991530627754978

## Gamma :0.01
Result on validation set

Accuracy : 0.7869565217391304, F1 Score :0.8171759995714225

## Gamma :0.1
Result on validation set

Accuracy : 0.10869565217391304, F1 Score :0.0196078431372549

# Best gama : 0.001

## Result on Test set
Accuracy : 0.9481481481481482, F1 Score :0.9472094045869722

## Result on Train set
Accuracy : 0.9992289899768697, F1 Score :0.9992337164750957