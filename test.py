import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.spatial import distance

# 1. Генерація випадкових даних
np.random.seed(58)
X = np.random.rand(1000, 1) * 100  # Випадкові числа в діапазоні 0-100
# y = np.random.rand(1000, 1) * 100
y = 3 * X.squeeze() + np.random.randn(1000) * 50  # Лінійна залежність з шумом

print("1. Згенеровані дані:")
print("Перші 5 значень X:", X[:5])
print("Перші 5 значень y:", y[:5])

# Відображення згенерованих даних до нормалізації
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.6)
plt.title('Згенеровані дані до нормалізації')
plt.xlabel('X')
plt.ylabel('y')
plt.grid(True)
plt.show()

# 2. Нормалізація даних
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("\n2. Нормалізовані дані:")
print("Перші 5 значень X після нормалізації:", X_scaled[:5])

# 3. Розділення на навчальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Відображення навчальної та тестової вибірок
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Навчальна вибірка', alpha=0.6)
plt.scatter(X_test, y_test, color='green', label='Тестова вибірка', alpha=0.6)
plt.title('Навчальна та Тестова вибірки')
plt.xlabel('Нормалізоване X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 4. Реалізація KNN вручну
def knn_predict(X_train, y_train, X_test, k):
    y_pred = []

    for test_point in X_test:
        # Обчислюємо відстані між поточною тестовою точкою і всіма тренувальними точками
        distances = [distance.euclidean(test_point, train_point) for train_point in X_train]

        # Знаходимо індекси k найближчих сусідів
        nearest_neighbors_indices = np.argsort(distances)[:k]

        # Вибираємо відповідні значення y тренувальної вибірки для найближчих сусідів
        nearest_neighbors_values = [y_train[i] for i in nearest_neighbors_indices]

        # Передбачення - середнє значення найближчих сусідів
        y_pred.append(np.mean(nearest_neighbors_values))

    return np.array(y_pred)


# Підбір оптимального значення K
k_values = range(1, 50)
errors = []

for k in k_values:
    y_pred = knn_predict(X_train, y_train, X_test, k)
    error = mean_squared_error(y_test, y_pred)
    errors.append(error)

# Вибір найкращого значення K
optimal_k = k_values[np.argmin(errors)]

# Візуалізація залежності помилки від кількості сусідів (K)
plt.figure(figsize=(10, 6))
plt.plot(k_values, errors, marker='o', color='orange', label='MSE для різних значень K')
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Найкраще K = {optimal_k}')
plt.title('Залежність помилки від кількості сусідів (K)')
plt.xlabel('K')
plt.ylabel('Середньоквадратична помилка (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# 5. Візуалізація отриманих рішень для найкращого K
y_pred_best = knn_predict(X_train, y_train, X_test, optimal_k)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Реальні значення')
plt.scatter(X_test, y_pred_best, color='red', label=f'Прогнозовані значення (K={optimal_k})')
plt.title('Реальні значення vs Прогнозовані значення (KNN)')
plt.xlabel('Нормалізоване X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

print("5. Середньоквадратична помилка для найкращого K:", mean_squared_error(y_test, y_pred_best))
