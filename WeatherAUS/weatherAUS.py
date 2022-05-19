import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

url = 'weatherAUS.csv'
data2 = pd.read_csv(url)
data2.drop(['Date', 'Sunshine', 'Evaporation', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm'], axis=1, inplace=True)
data2.drop(['Location', 'MinTemp', 'MaxTemp', 'WindSpeed9am', 'WindSpeed3pm'], axis=1, inplace=True)


data2.RainTomorrow.replace(['No', 'Yes'], [0, 1], inplace=True)
data2.RainToday.replace(['No', 'Yes'], [0, 1], inplace=True)
#data2.WindGustDir.replace(['no', 'yes'], [0, 1], inplace=True)
data2.WindGustDir.replace(['W', 'WNW', 'WSW', 'NE', 'E', 'NNW', 'N', 'SE', 'NNE', 'ENE', 'SSE', 'SW', 'SSW', 'ESE', 'NW', 'S'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], inplace=True)
data2.WindDir9am.replace(['W', 'WNW', 'WSW', 'NE', 'E', 'NNW', 'N', 'SE', 'NNE', 'ENE', 'SSE', 'SW', 'SSW', 'ESE', 'NW', 'S'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], inplace=True)
data2.WindDir3pm.replace(['W', 'WNW', 'WSW', 'NE', 'E', 'NNW', 'N', 'SE', 'NNE', 'ENE', 'SSE', 'SW', 'SSW', 'ESE', 'NW', 'S'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], inplace=True)
data2.dropna(axis=0,how='any', inplace=True)

#dividimos los datos en dos
data2_train = data2[:71096]
data2_test = data2[71096:]

x = np.array(data2_train.drop(['RainTomorrow'], 1))
y = np.array(data2_train.RainTomorrow) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


x_test_out = np.array(data2_test.drop(['RainTomorrow'], 1))
y_test_out = np.array(data2_test.RainTomorrow) # 0 no acepto, 1 si acepto
# Regresión Logística


# Seleccionar un modelo
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entreno el modelo
logreg.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')

# MAQUINA DE SOPORTE VECTORIAL

# Seleccionar un modelo
svc = SVC(gamma='auto')

# Entreno el modelo
svc.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')

# ARBOL DE DECISIÓN

# Seleccionar un modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')

#REGRESION LINEAL

#seleccionar un modelo
rl = LinearRegression()

#entreamo el modelo
rl.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Regresion Lineal')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {rl.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {rl.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {rl.score(x_test_out, y_test_out)}')

#RANDOM FOREST

#seleccionar un modelo
rf = RandomForestClassifier()

#entreamo el modelo
rf.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Random Forest')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {rf.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {rf.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {rf.score(x_test_out, y_test_out)}')
