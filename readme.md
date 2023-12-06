Paper:https://aclanthology.org/2023.ccl-1.63/


# 实验设计
## Early Fusion
### 控制后续模型以及分类器不变，只改变句子特征的产生方式（Feature Fusion）
### 拼接方式：Mul and Concat
### Pooling方式：Max Mean Sum
##
## Model Fusion
### 控制分类器与拼接方式不变
### Gate BiLSTM
##
## Late Fusion (Decision Fusion)
