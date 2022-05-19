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
data.y.replace(["yes", "no"], [1, 0], inplace=True)
data.education.replace(
    ["unknown", "secondary", "primary", "tertiary"], [0, 1, 2, 3], inplace=True
)

print(data.loan.value_counts())

age_mean = data.age.mean()
data.age.replace(np.nan, age_mean, inplace=True)


num_of_rows = data.shape[0]
train_size = int(num_of_rows * 0.8)
test_size = num_of_rows - train_size

train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

x = np.array(train_data.drop(["y"], axis=1))
y = np.array(train_data["y"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test_out = np.array(test_data.drop(["y"], axis=1))
y_test_out = np.array(test_data["y"])

print("Models:")

log_reg = LogisticRegression(solver="lbfgs", max_iter=train_size)
log_reg.fit(x_train, y_train)
log_reg_accuracy = log_reg.score(x_test, y_test)
log_reg_accuracy_validation = log_reg.score(x_test_out, y_test_out)
print("------------------ Logistic Regression ------------------")
print("Accuracy           : ", log_reg_accuracy)
print("Accuracy validation: ", log_reg_accuracy_validation)

svc = SVC(gamma="auto")
svc.fit(x_train, y_train)
svc_accuracy = svc.score(x_test, y_test)
svc_accuracy_validation = svc.score(x_test_out, y_test_out)
print("------------------ Support Vector Machine ------------------")
print("Accuracy           : ", svc_accuracy)
print("Accuracy validation: ", svc_accuracy_validation)

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
tree_accuracy = tree.score(x_test, y_test)
tree_validation = tree.score(x_test_out, y_test_out)
print("------------------ Decision Tree ------------------")
print("Accuracy           : ", tree_accuracy)
print("Accuracy validation: ", tree_validation)

kn_classifier = KNeighborsClassifier()
kn_classifier.fit(x_train, y_train)
kn_classifier_accuracy = kn_classifier.score(x_test, y_test)
kn_classifier_validation = kn_classifier.score(x_test_out, y_test_out)
print("------------------ K-Nearest Neighbors ------------------")
print("Accuracy           : ", kn_classifier_accuracy)
print("Accuracy validation: ", kn_classifier_validation)

random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)
random_forest_accuracy = random_forest.score(x_test, y_test)
random_forest_validation = random_forest.score(x_test_out, y_test_out)
print("------------------ Random Forest ------------------")
print("Accuracy           : ", random_forest_accuracy)
print("Accuracy validation: ", random_forest_validation)

print("\n\n")
print("Cross Validation:")

kfold = KFold(n_splits=10)
acc_scores_train_train = []
acc_scores_train_test = []
for train, test in kfold.split(x, y):
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]
    log_reg.fit(x_train, y_train)
    acc_scores_train_train.append(log_reg.score(x_train, y_train))
    acc_scores_train_test.append(log_reg.score(x_test, y_test))

y_pred = log_reg.predict(x_test_out)
train_accuracy = np.mean(acc_scores_train_train)
test_accuracy = np.mean(acc_scores_train_test)
validation_accuracy = log_reg.score(x_test_out, y_test_out)
recall = recall_score(y_test_out, y_pred)
precision = precision_score(y_test_out, y_pred)
f1 = f1_score(y_test_out, y_pred)
matrix = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(matrix)
plt.title("Confusion Matrix for Logistic Regression")
plt.show()
print("------------------ Logistic Regression ------------------")
print("Accuracy train-train : ", np.mean(acc_scores_train_train))
print("Accuracy train-test  : ", np.mean(acc_scores_train_test))
print("Train accuracy       : ", train_accuracy)
print("Test accuracy        : ", test_accuracy)
print("Validation accuracy  : ", validation_accuracy)
print("Recall               : ", recall)
print("Precision            : ", precision)
print("F1 score             : ", f1)
print("Real                 : ", y_test_out)
print("Predicted            : ", y_pred)

kfold = KFold(n_splits=10)
acc_scores_train_train = []
acc_scores_train_test = []
for train, test in kfold.split(x):
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]
    svc.fit(x_train, y_train)
    acc_scores_train_train.append(svc.score(x_train, y_train))
    acc_scores_train_test.append(svc.score(x_test, y_test))

y_pred = svc.predict(x_test_out)
train_accuracy = np.mean(acc_scores_train_train)
test_accuracy = np.mean(acc_scores_train_test)
validation_accuracy = svc.score(x_test_out, y_test_out)
recall = recall_score(y_test_out, y_pred)
precision = precision_score(y_test_out, y_pred)
f1 = f1_score(y_test_out, y_pred)
matrix = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(matrix)
plt.title("Confusion Matrix - SVC")
plt.show()
print("------------------ Support Vector Machine ------------------")
print("Accuracy train-train: ", np.mean(acc_scores_train_train))
print("Accuracy train-test : ", np.mean(acc_scores_train_test))
print("Train accuracy      : ", train_accuracy)
print("Test accuracy       : ", test_accuracy)
print("Validation accuracy : ", validation_accuracy)
print("Recall              : ", recall)
print("Precision           : ", precision)
print("F1 score            : ", f1)
print("Real                 : ", y_test_out)
print("Predicted            : ", y_pred)

kfold = KFold(n_splits=10)
acc_scores_train_train = []
acc_scores_train_test = []
for train, test in kfold.split(x):
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]
    tree.fit(x_train, y_train)
    acc_scores_train_train.append(tree.score(x_train, y_train))
    acc_scores_train_test.append(tree.score(x_test, y_test))

y_pred = tree.predict(x_test_out)
train_accuracy = np.mean(acc_scores_train_train)
test_accuracy = np.mean(acc_scores_train_test)
validation_accuracy = tree.score(x_test_out, y_test_out)
recall = recall_score(y_test_out, y_pred)
precision = precision_score(y_test_out, y_pred)
f1 = f1_score(y_test_out, y_pred)
matrix = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(matrix)
plt.title("Confusion Matrix (Decision Tree)")
plt.show()
print("------------------ Decision Tree ------------------")
print("Accuracy train-train: ", np.mean(acc_scores_train_train))
print("Accuracy train-test : ", np.mean(acc_scores_train_test))
print("Train accuracy      : ", train_accuracy)
print("Test accuracy       : ", test_accuracy)
print("Validation accuracy : ", validation_accuracy)
print("Recall              : ", recall)
print("Precision           : ", precision)
print("F1 score            : ", f1)
print("Real                 : ", y_test_out)
print("Predicted            : ", y_pred)


kfold = KFold(n_splits=10)
acc_scores_train_train = []
acc_scores_train_test = []
for train, test in kfold.split(x):
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]
    kn_classifier.fit(x_train, y_train)
    acc_scores_train_train.append(kn_classifier.score(x_train, y_train))
    acc_scores_train_test.append(kn_classifier.score(x_test, y_test))

y_pred = kn_classifier.predict(x_test_out)
train_accuracy = np.mean(acc_scores_train_train)
test_accuracy = np.mean(acc_scores_train_test)
validation_accuracy = kn_classifier.score(x_test_out, y_test_out)
recall = recall_score(y_test_out, y_pred)
precision = precision_score(y_test_out, y_pred)
f1 = f1_score(y_test_out, y_pred)
matrix = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(matrix)
plt.title("Confusion Matrix - KNN")
plt.show()
print("------------------ K-Nearest Neighbors ------------------")
print("Accuracy train-train: ", np.mean(acc_scores_train_train))
print("Accuracy train-test : ", np.mean(acc_scores_train_test))
print("Train accuracy      : ", train_accuracy)
print("Test accuracy       : ", test_accuracy)
print("Validation accuracy : ", validation_accuracy)
print("Recall              : ", recall)
print("Precision           : ", precision)
print("F1 score            : ", f1)
print("Real                 : ", y_test_out)
print("Predicted            : ", y_pred)

kfold = KFold(n_splits=10)
acc_scores_train_train = []
acc_scores_train_test = []
for train, test in kfold.split(x):
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]
    random_forest.fit(x_train, y_train)
    acc_scores_train_train.append(random_forest.score(x_train, y_train))
    acc_scores_train_test.append(random_forest.score(x_test, y_test))

y_pred = random_forest.predict(x_test_out)
train_accuracy = np.mean(acc_scores_train_train)
test_accuracy = np.mean(acc_scores_train_test)
validation_accuracy = random_forest.score(x_test_out, y_test_out)
recall = recall_score(y_test_out, y_pred)
precision = precision_score(y_test_out, y_pred)
f1 = f1_score(y_test_out, y_pred)
matrix = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(matrix)
plt.title("Confusion Matrix - Random Forest")
plt.show()
print("------------------ Random Forest ------------------")
print("Accuracy train-train: ", np.mean(acc_scores_train_train))
print("Accuracy train-test : ", np.mean(acc_scores_train_test))
print("Train accuracy      : ", train_accuracy)
print("Test accuracy       : ", test_accuracy)
print("Validation accuracy : ", validation_accuracy)
print("Recall              : ", recall)
print("Precision           : ", precision)
print("F1 score            : ", f1)
print("Real                 : ", y_test_out)
print("Predicted            : ", y_pred)
