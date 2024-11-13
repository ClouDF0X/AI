import numpy as np
import matplotlib.pyplot as plt


# Функція втрат (Mean Squared Error)
def compute_loss(X, y, k, b):
    m = len(y)
    predictions = X.dot(k) + b
    loss = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return loss


# Градієнтний спуск
def gradient_descent(X, y, k, b, learning_rate, iterations):
    m = len(y)

    for i in range(iterations):
        # Обчислення прогнозів
        predictions = X.dot(k) + b

        # Обчислення градієнтів
        dk = (1 / m) * X.T.dot(predictions - y)
        db = (1 / m) * np.sum(predictions - y)

        # Оновлення параметрів
        k = k - learning_rate * dk
        b = b - learning_rate * db

        # Вивід значення функції втрат кожні 100 ітерацій
        if i % 100 == 0:
            loss = compute_loss(X, y, k, b)
            print(f"Iteration {i}: Loss = {loss}")

    return k, b


# Генерація випадкових даних для лінійної регресії
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Початкові значення параметрів
k = np.zeros((X.shape[1], 1))  # бо одна ознака
b = 0

# Налаштування параметрів градієнтного спуску
learning_rate = 0.01
iterations = 1000

# Виконання градієнтного спуску
k, b = gradient_descent(X, y, k, b, learning_rate, iterations)

# Вивід оптимізованих параметрів
final_loss = compute_loss(X, y, k, b)
print(f"\nОптимізовані параметри: k = {k.flatten()}, b = {b}")
print(f"Похибка (функція втрат) з оптимізованими параметрами: {final_loss}")

# Побудова графіка з даними без лінії регресії
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Дані')
plt.title('Точки даних для лінійної регресії')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Дані')
plt.plot(X, X.dot(k) + b, color='red', label='Лінія регресії')
plt.title('Лінійна регресія з використанням градієнтного спуску')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


