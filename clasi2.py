import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# Cargar el archivo CSV
file_path = 'categorias.csv'
df = pd.read_csv(file_path)

# Graficar la distribución de clases
class_counts = df['is_dead'].value_counts()

# Crear un gráfico de barras
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Classes')
plt.xlabel('Class (0: Alive, 1: Dead)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Separar las características (X) y la variable objetivo (y)
X = df.drop(columns=['is_dead', 'categoria_edad'])
y = df['is_dead']

# Realizar la partición estratificada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Definir el modelo de Random Forest
clf = RandomForestClassifier(random_state=42)

# Definir los parámetros para la búsqueda de cuadrícula
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Realizar la búsqueda de cuadrícula
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Obtener los mejores hiperparámetros
best_params = grid_search.best_params_
print(f'Mejores hiperparámetros: {best_params}')

# Ajustar el modelo con los mejores hiperparámetros
best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
y_pred = best_clf.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy del Random Forest con los mejores hiperparámetros sobre el conjunto de prueba: {accuracy:.2f}')

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Alive', 'Dead'], yticklabels=['Alive', 'Dead'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Calcular Precision, Recall y F1-Score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Imprimir las métricas
print(f'Precision del Random Forest sobre el conjunto de prueba: {precision:.2f}')
print(f'Recall del Random Forest sobre el conjunto de prueba: {recall:.2f}')
print(f'F1-Score del Random Forest sobre el conjunto de prueba: {f1:.2f}')

# Comparar el F1-Score con el accuracy
print(f'Comparación - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}')
