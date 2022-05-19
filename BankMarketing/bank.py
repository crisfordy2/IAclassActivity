from re import X
from warnings import simplefilter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

simplefilter(action='ignore', category=FutureWarning)

url = 'bank-full.csv'
data = pd.read_csv(url)

# Data treatment
# replace education with numbers
data.education.replace(['primary', 'secondary', 'tertiary', 'unknown'], [
                       0, 1, 2, 3], inplace=True)

# replace job with numbers
data.job.replace(['admin.', 'blue-collar', 'entrepreneur', 'housemaid',
                  'management', 'retired', 'self-employed', 'services',
                  'student', 'technician', 'unemployed', 'unknown'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)

data.loan.replace(['no', 'yes'], [0, 1], inplace=True)
data.contact.replace(['cellular', 'telephone', 'unknown'], [
                     0, 1, 2], inplace=True)

data.month.replace(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
                   'oct', 'nov', 'dec'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)

# replace poutcome with numbers
data.poutcome.replace(['failure', 'success', 'other', 'unknown'], [
                      0, 1, 2, 3], inplace=True)
data.housing.replace(['no', 'yes'], [0, 1], inplace=True)
data.default.replace(['no', 'yes'], [0, 1], inplace=True)
data.marital.replace(['divorced', 'married', 'single'],
                     [0, 1, 2], inplace=True)
data.y.replace(['no', 'yes'], [0, 1], inplace=True)
data.drop(['campaign', 'pdays', 'previous'], axis=1, inplace=True)

data.age.replace(np.nan, 40, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 70, 100]
names = ['1', '2', '3', '4', '5', '6', '7', '8']
data.age = pd.cut(data.age, rangos, labels=names)

data.dropna(axis=0, how='any', inplace=True)

# Break the data in two
data_train = data[:20000]
data_test = data[20000:]

# print(data_train)

# Split the data
x = np.array(data_train.drop(['education'], 1))
y = np.array(data_train.education)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x_test_out = np.array(data_test.drop(['education'], 1))
y_test_out = np.array(data_test.education)


# Selecting a model
logisticReg = LogisticRegression(solver='lbfgs', max_iter=7600)

# Actual train of the model
logisticReg.fit(x_train, y_train)

# Metrics
print(
    f'Accuracy de Entrenamiento con Valores de Entrenamiento: {logisticReg.score(x_train, y_train)}')

print(
    f'Accuracy de Test con Valores de Test: {logisticReg.score(x_test, y_test)}')

print(f'Accuracy de Validación: {logisticReg.score(x_test_out, y_test_out)}')


# Decision Tree

# Seleccionar un modelo
tree = DecisionTreeClassifier()

# Entreno el modelo
tree.fit(x_train, y_train)

# Other metrics

print('---------------------Decisión Tree---------------------')

print(
    f'Accuracy de Entrenamiento con valores de Entrenamiento: {tree.score(x_train, y_train)}')

print(
    f'Accuracy de Test con valores de Entrenamiento: {tree.score(x_test, y_test)}')

print(f'Accuracy de Validación: {tree.score(x_test_out, y_test_out)}')

# SVC

# Selecting the model
svc = SVC(gamma='auto')

# Train the model
svc.fit(x_train, y_train)

print('---------------------SVC---------------------')

# Accuracy de Entrenamiento de Entrenamiento
print(
    f'Accuracy de Entrenamiento con valores de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(
    f'Accuracy de Test con valores de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'Accuracy de Validación: {svc.score(x_test_out, y_test_out)}')