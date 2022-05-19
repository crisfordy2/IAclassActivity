from warnings import simplefilter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

simplefilter(action='ignore', category=FutureWarning)

url = 'diabetes.csv'
data = pd.read_csv(url)

print(data.Age.mean())

# Adding ranges for age
data.Age.replace(np.nan, 33, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 50]
nombres = ['1', '2', '3', '4', '5', '6']
data.Age = pd.cut(data.Age, rangos, labels=nombres)

# Remove some fields
data.drop(['Pregnancies', 'DiabetesPedigreeFunction',
          'SkinThickness'], axis=1, inplace=True)

data.dropna(axis=0, how='any', inplace=True)

# Break the data in two
data_train = data[:344]
data_test = data[343:]

# Split the data
x = np.array(data_train.drop(['Outcome'], axis=1))
y = np.array(data_test.Outcome)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome)

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