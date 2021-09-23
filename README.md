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


### Train size: 1257, Test size: 270, Val size: 270

## Gamma :1e-07
Result on validation set

Accuracy : 0.1, F1 Score :0.01818181818181818

## Gamma :1e-06
Result on validation set

Accuracy : 0.1, F1 Score :0.01818181818181818

## Gamma :1e-05
Result on validation set

Accuracy : 0.8740740740740741, F1 Score :0.8732673548597052

## Gamma :0.0001
Result on validation set

Accuracy : 0.9555555555555556, F1 Score :0.9545831499566034

## Gamma :0.001
Result on validation set

Accuracy : 0.9925925925925926, F1 Score :0.9926484660849987

## Gamma :0.01
Result on validation set

Accuracy : 0.7555555555555555, F1 Score :0.7954831950066726

## Gamma :0.1
Result on validation set

Accuracy : 0.1, F1 Score :0.01818181818181818

# Best gama : 0.001

## Result on Test set
Accuracy : 0.9481481481481482, F1 Score :0.9477475781671046

## Result on Train set
Accuracy : 0.9992044550517104, F1 Score :0.9992031872509962