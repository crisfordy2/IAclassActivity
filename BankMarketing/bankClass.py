import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from matplotlib.pyplot import axis
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("bank-full.csv")

data.drop(
    [
        "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"
    ],
    axis=1,
    inplace=True,
)

data.dropna(axis=0, inplace=True, how="any")

data.loan.replace(["yes", "no"], [1, 0], inplace=True)
data.housing.replace(["yes", "no"], [1, 0], inplace=True)
data.infoY.replace(["yes", "no"], [1, 0], inplace=True)
data.education.replace(
    ["unknown", "secondary", "primary", "tertiary"], [0, 1, 2, 3], inplace=True
)

print(data.loan.value_counts())

age_mean = data.age.mean()
data.age.replace(np.nan, age_mean, inplace=True)


numRows = data.shape[0]
trainignSize = int(numRows * 0.8)
test_size = numRows - trainingSize

train_data = data.iloc[:trainingSize]
test_data = data.iloc[trainingSize:]

infoX = np.array(train_data.drop(["infoY"], axis=1))
infoY = np.array(train_data["infoY"])
trainingX, dataTestX, trainingDataY, y_test = train_test_split(
    infoX, infoY, test_size=0.2)
testOutX = np.array(test_data.drop(["infoY"], axis=1))
testOutY = np.array(test_data["infoY"])


log_reg = LogisticRegression(solver="lbfgs", max_iter=trainingSize)
log_reg.fit(trainingX, trainingDataY)
log_reg_accuracy = log_reg.score(dataTestX, y_test)
log_reg_accuracy_validation = log_reg.score(testOutX, testOutY)


kfold = KFold(n_splits=10)
scoresAccTraining = []
scoresAccTest = []
for train, test in kfold.split(infoX, infoY):
    trainingX, dataTestX = infoX[train], infoX[test]
    trainingDataY, y_test = infoY[train], infoY[test]
    log_reg.fit(trainingX, trainingDataY)
    scoresAccTraining.append(log_reg.score(trainingX, trainingDataY))
    scoresAccTest.append(log_reg.score(dataTestX, y_test))

y_pred = log_reg.predict(testOutX)
train_accuracy = np.mean(scoresAccTraining)
test_accuracy = np.mean(scoresAccTest)
validation_accuracy = log_reg.score(testOutX, testOutY)
recall = recall_score(testOutY, y_pred)
precision = precision_score(testOutY, y_pred)
f1 = f1_score(testOutY, y_pred)
matrix = confusion_matrix(testOutY, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(matrix)


kfold = KFold(n_splits=10)
scoresAccTraining = []
scoresAccTest = []
for train, test in kfold.split(infoX):
    trainingX, dataTestX = infoX[train], infoX[test]
    trainingDataY, y_test = infoY[train], infoY[test]
    svc.fit(trainingX, trainingDataY)
    scoresAccTraining.append(svc.score(trainingX, trainingDataY))
    scoresAccTest.append(svc.score(dataTestX, y_test))

y_pred = svc.predict(testOutX)
train_accuracy = np.mean(scoresAccTraining)
test_accuracy = np.mean(scoresAccTest)
validation_accuracy = svc.score(testOutX, testOutY)
recall = recall_score(testOutY, y_pred)
precision = precision_score(testOutY, y_pred)
f1 = f1_score(testOutY, y_pred)
matrix = confusion_matrix(testOutY, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(matrix)


kfold = KFold(n_splits=10)
scoresAccTraining = []
scoresAccTest = []
for train, test in kfold.split(infoX):
    trainingX, dataTestX = infoX[train], infoX[test]
    trainingDataY, y_test = infoY[train], infoY[test]
    tree.fit(trainingX, trainingDataY)
    scoresAccTraining.append(tree.score(trainingX, trainingDataY))
    scoresAccTest.append(tree.score(dataTestX, y_test))

y_pred = tree.predict(testOutX)
train_accuracy = np.mean(scoresAccTraining)
test_accuracy = np.mean(scoresAccTest)
validation_accuracy = tree.score(testOutX, testOutY)
recall = recall_score(testOutY, y_pred)
precision = precision_score(testOutY, y_pred)
f1 = f1_score(testOutY, y_pred)
matrix = confusion_matrix(testOutY, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(matrix)


kfold = KFold(n_splits=10)
scoresAccTraining = []
scoresAccTest = []
for train, test in kfold.split(infoX):
    trainingX, dataTestX = infoX[train], infoX[test]
    trainingDataY, y_test = infoY[train], infoY[test]
    kn_classifier.fit(trainingX, trainingDataY)
    scoresAccTraining.append(kn_classifier.score(trainingX, trainingDataY))
    scoresAccTest.append(kn_classifier.score(dataTestX, y_test))

y_pred = kn_classifier.predict(testOutX)
train_accuracy = np.mean(scoresAccTraining)
test_accuracy = np.mean(scoresAccTest)
validation_accuracy = kn_classifier.score(testOutX, testOutY)
recall = recall_score(testOutY, y_pred)
precision = precision_score(testOutY, y_pred)
f1 = f1_score(testOutY, y_pred)
matrix = confusion_matrix(testOutY, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(matrix)


kfold = KFold(n_splits=10)
scoresAccTraining = []
scoresAccTest = []
for train, test in kfold.split(infoX):
    trainingX, dataTestX = infoX[train], infoX[test]
    trainingDataY, y_test = infoY[train], infoY[test]
    random_forest.fit(trainingX, trainingDataY)
    scoresAccTraining.append(random_forest.score(trainingX, trainingDataY))
    scoresAccTest.append(random_forest.score(dataTestX, y_test))

y_pred = random_forest.predict(testOutX)
train_accuracy = np.mean(scoresAccTraining)
test_accuracy = np.mean(scoresAccTest)
validation_accuracy = random_forest.score(testOutX, testOutY)
recall = recall_score(testOutY, y_pred)
precision = precision_score(testOutY, y_pred)
f1 = f1_score(testOutY, y_pred)
matrix = confusion_matrix(testOutY, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(matrix)


print("################## Regresión logística ##################")
print("Accuracy           : ", log_reg_accuracy)
print("Accuracy validation: ", log_reg_accuracy_validation)

svc = SVC(gamma="auto")
svc.fit(trainingX, trainingDataY)
svc_accuracy = svc.score(dataTestX, y_test)
svc_accuracy_validation = svc.score(testOutX, testOutY)
print("################## Support Vector Machine ##################")
print("Accuracy           : ", svc_accuracy)
print("Accuracy validation: ", svc_accuracy_validation)

tree = DecisionTreeClassifier()
tree.fit(trainingX, trainingDataY)
tree_accuracy = tree.score(dataTestX, y_test)
tree_validation = tree.score(testOutX, testOutY)
print("################## Decision Tree ##################")
print("Accuracy           : ", tree_accuracy)
print("Accuracy validation: ", tree_validation)

kn_classifier = KNeighborsClassifier()
kn_classifier.fit(trainingX, trainingDataY)
kn_classifier_accuracy = kn_classifier.score(dataTestX, y_test)
kn_classifier_validation = kn_classifier.score(testOutX, testOutY)
print("###################  KNearest Neighbors ##################")
print("Accuracy           : ", kn_classifier_accuracy)
print("Accuracy validation: ", kn_classifier_validation)

random_forest = RandomForestClassifier()
random_forest.fit(trainingX, trainingDataY)
random_forest_accuracy = random_forest.score(dataTestX, y_test)
random_forest_validation = random_forest.score(testOutX, testOutY)
print("################## Random Forest ##################")
print("Accuracy           : ", random_forest_accuracy)
print("Accuracy validation: ", random_forest_validation)

print("\n\n")
print("Validation: Cross ")

print("Models:")

plt.title("Confusion Matrix for Logistic Regression")
plt.show()
print("------------------ Logistic Regression ------------------")
print("Accuracy train-train : ", np.mean(scoresAccTraining))
print("Accuracy train-test  : ", np.mean(scoresAccTest))
print("Train accuracy       : ", train_accuracy)
print("Test accuracy        : ", test_accuracy)
print("Validation accuracy  : ", validation_accuracy)
print("Recall               : ", recall)
print("Precision            : ", precision)
print("F1 score             : ", f1)
print("Real                 : ", testOutY)
print("Predicted            : ", y_pred)


plt.title("Confusion Matrix - SVC")
plt.show()
print("------------------ Support Vector Machine ------------------")
print("Accuracy train-train: ", np.mean(scoresAccTraining))
print("Accuracy train-test : ", np.mean(scoresAccTest))
print("Train accuracy      : ", train_accuracy)
print("Test accuracy       : ", test_accuracy)
print("Validation accuracy : ", validation_accuracy)
print("Recall              : ", recall)
print("Precision           : ", precision)
print("F1 score            : ", f1)
print("Real                 : ", testOutY)
print("Predicted            : ", y_pred)


plt.title("Confusion Matrix (Decision Tree)")
plt.show()
print("------------------ Decision Tree ------------------")
print("Accuracy train-train: ", np.mean(scoresAccTraining))
print("Accuracy train-test : ", np.mean(scoresAccTest))
print("Train accuracy      : ", train_accuracy)
print("Test accuracy       : ", test_accuracy)
print("Validation accuracy : ", validation_accuracy)
print("Recall              : ", recall)
print("Precision           : ", precision)
print("F1 score            : ", f1)
print("Real                 : ", testOutY)
print("Predicted            : ", y_pred)


plt.title("Confusion Matrix - KNN")
plt.show()
print("------------------ K-Nearest Neighbors ------------------")
print("Accuracy train-train: ", np.mean(scoresAccTraining))
print("Accuracy train-test : ", np.mean(scoresAccTest))
print("Train accuracy      : ", train_accuracy)
print("Test accuracy       : ", test_accuracy)
print("Validation accuracy : ", validation_accuracy)
print("Recall              : ", recall)
print("Precision           : ", precision)
print("F1 score            : ", f1)
print("Real                 : ", testOutY)
print("Predicted            : ", y_pred)


plt.title("Confusion Matrix - Random Forest")
plt.show()
print("------------------ Random Forest ------------------")
print("Accuracy train-train: ", np.mean(scoresAccTraining))
print("Accuracy train-test : ", np.mean(scoresAccTest))
print("Train accuracy      : ", train_accuracy)
print("Test accuracy       : ", test_accuracy)
print("Validation accuracy : ", validation_accuracy)
print("Recall              : ", recall)
print("Precision           : ", precision)
print("F1 score            : ", f1)
print("Real                 : ", testOutY)
print("Predicted            : ", y_pred)
