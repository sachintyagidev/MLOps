# MLOps
## Test cases with pytest:

Command to execute from root of repo: 

pytest -s

# Train size: 1257, Test size: 270, Val size: 270

### SVM Gama : 1e-07, Accuracy : 0.1, F1 : 0.01818181818181818
### SVM Gama : 1e-06, Accuracy : 0.1, F1 : 0.01818181818181818
### SVM Gama : 1e-05, Accuracy : 0.8740740740740741, F1 : 0.8732673548597052
### SVM Gama : 0.0001, Accuracy : 0.9555555555555556, F1 : 0.9545831499566034
### SVM Gama : 0.001, Accuracy : 0.9925925925925926, F1 : 0.9926484660849987
### SVM Gama : 0.01, Accuracy : 0.7555555555555555, F1 : 0.7954831950066726

## Mean : 0.6296296296296297 and standard deviations 0.38179299279339357 of accuracy for model.

# Best Gama Value: 0.001

### DT Depth : 2, Accuracy : 0.3296296296296296, F1 : 0.23444641674163985
### DT Depth : 4, Accuracy : 0.6074074074074074, F1 : 0.5310884458053379
### DT Depth : 6, Accuracy : 0.7592592592592593, F1 : 0.7528688350586267
### DT Depth : 8, Accuracy : 0.762962962962963, F1 : 0.7511478137991315
### DT Depth : 10, Accuracy : 0.8148148148148148, F1 : 0.8131148932189344
### DT Depth : 12, Accuracy : 0.8222222222222222, F1 : 0.822085732507752
### DT Depth : 14, Accuracy : 0.8296296296296296, F1 : 0.8293442828107815
### DT Depth : 16, Accuracy : 0.8111111111111111, F1 : 0.8080815586647703
## Mean : 0.7171296296296297 and standard deviations 0.16130190384435467 of accuracy for model.

# Best Depth Value: 14.0

# Decision tree Mean accuracy is better and deviation is also less, so the decision tree model is more stable to depth change in comparison to SVM.