import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Завантажуємо iris dataset
iris = datasets.load_iris()

# Вибираємо дві ознаки: sepal length і sepal width
feature1 = 0
feature2 = 1

# Формуємо новий датасет з обраними ознаками та класами
X = iris.data[:, [feature1, feature2]]
y = iris.target

# Побудова графіку
plt.figure(figsize=(10, 6))
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=target_name)
plt.xlabel(iris.feature_names[feature1])
plt.ylabel(iris.feature_names[feature2])
plt.legend()
plt.title('Iris Dataset')
plt.show()

# Нормалізуємо параметри квітів
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Побудова графіку
plt.figure(figsize=(10, 6))
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=target_name)
plt.xlabel(iris.feature_names[feature1])
plt.ylabel(iris.feature_names[feature2])
plt.legend()
plt.title('Iris Dataset')
plt.show()

# Розділяємо дані на навчальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)


# Побудова графіку для навчальної та тестової вибірок на одному графіку
plt.figure(figsize=(10, 6))

# Відображаємо навчальну вибірку з прозорістю
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], label=f'{target_name} (Train)', alpha=0.5)

# Відображаємо тестову вибірку
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1], label=f'{target_name} (Test)', alpha=1.0)

plt.xlabel(iris.feature_names[feature1])
plt.ylabel(iris.feature_names[feature2])
plt.legend()
plt.title('Навчальні та тестові вибірки Iris Dataset')
plt.grid()
plt.show()

# Список для збереження результатів
k_values = list(range(1, 40))
accuracies = []

# Перебираємо різні значення К
for k in k_values:
    # Створюємо і навчаємо KNN-класифікатор
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Передбачаємо на тестовій вибірці
    y_pred = knn.predict(X_test)

    # Обчислюємо точність класифікації
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Знаходимо найкраще значення К
best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)

print(f'Найкраще значення K: {best_k}')
print(f'Найвища точність: {best_accuracy:.4f}')

# Відображаємо точність класифікації для різних значень K
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('Значення K')
plt.ylabel('Точність класифікації')
plt.title('Залежність точності класифікації від K для KNN')
plt.grid()
plt.show()


# Використовуємо KNN з найкращим K для прогнозування
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred_best = knn_best.predict(X_test)

# Відображення тестових даних з передбаченнями
plt.figure(figsize=(10, 6))

# Визначаємо маркери для кожного фактичного класу
markers = ['x', 's', '^']  # Круги, квадрати, трикутники
colors = ['blue', 'green', 'orange']  # Один колір для всіх фактичних класів

# Відображаємо тестову вибірку з різними маркерами
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1],
                label=f'Фактичний {target_name}', alpha=0.7, marker=markers[i], color=colors[i])

# Додаємо передбачення до графіку з іншими кольорами
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_test[y_pred_best == i, 0], X_test[y_pred_best == i, 1],
                label=f'Передбачено {target_name} (K={best_k})', edgecolor='black', alpha=0.2, marker='o', color=colors[i])

# Визначаємо точки, для яких фактичний клас не співпадає з передбаченим
misclassified_indices = np.where(y_pred_best != y_test)[0]

if len(misclassified_indices) > 0:
    plt.scatter(X_test[misclassified_indices, 0], X_test[misclassified_indices, 1],
                label='Неправильно передбачені', edgecolor='red', alpha=0.9, marker='x', s=100)

plt.xlabel(iris.feature_names[feature1])
plt.ylabel(iris.feature_names[feature2])
plt.legend()
plt.title(f'Тестові дані з передбаченнями KNN (K={best_k})')
plt.grid()
plt.show()