# import requests
# from bs4 import BeautifulSoup
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import time
# import csv
#
#
# def check_number(number):
#     url = f"https://www.livemint.com/search?q={number}"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#
#     result_elements = soup.find_all('div', class_='result-item')
#     return number, len(result_elements) > 0
#
#
# numbers_to_check = ["123456", "789012", "345678"]
# # numbers_to_check = list(map(lambda x: x + 1, range(100000, 1000000)))
#
# with open('results.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Number', 'Found'])
#
# start_time = time.time()
#
# with ThreadPoolExecutor(max_workers=10) as executor:
#     future_to_number = {executor.submit(check_number, number): number for number in numbers_to_check}
#
#     for future in as_completed(future_to_number):
#         try:
#             number, found = future.result()
#             print(f"{number}: {'Найден' if found else 'Не найден'}")
#             writer.writerow([number, found])
#         except Exception as e:
#             print(f"{future_to_number[future]} generated an exception: {e}")
#
# end_time = time.time()
# print(f"Time taken: {end_time - start_time} seconds")


# import matplotlib.pyplot as plt
# import numpy as np
#
# # Создаем тестовые данные
# x = np.array([1, 2, 3])
# y = np.array([1, 4, 2])
# colors = ['red', 'green', 'blue']  # Задаем разные цвета для точек
#
# # Создаем график
# scatter = plt.scatter(x, y, c=colors)
#
# # Получаем объект Collection
# collection = scatter.get_facecolors()
#
#
# # Функция для поиска цвета точки по координатам
# def get_point_color(x_target, y_target, x_coords, y_coords, tolerance=0.1):
#     # Находим индекс ближайшей точки
#     distances = np.sqrt((x_coords - x_target) ** 2 + (y_coords - y_target) ** 2)
#     min_distance_idx = np.argmin(distances)
#
#     # Проверяем, находится ли точка в пределах допустимого расстояния
#     if distances[min_distance_idx] <= tolerance:
#         return collection[min_distance_idx]
#     return None
#
#
# # Пример использования
# target_x, target_y = 2, 4  # Ищем цвет точки рядом с этими координатами
# point_color = get_point_color(target_x, target_y, x, y)
#
# if point_color is not None:
#     print(f"Цвет найденной точки: {point_color}")
# else:
#     print("Точка не найдена в пределах допустимой погрешности")
#


import matplotlib.pyplot as plt
import numpy as np

# Создаем тестовые данные
x = np.array([1, 2, 3])
y = np.array([1, 4, 2])
colors = ['red', 'green', 'blue']

# Создаем график
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x, y, c=colors, s=100)


# Функция для поиска цвета точки
def get_point_color(x_target, y_target, x_coords, y_coords, tolerance=0.1):
    distances = np.sqrt((x_coords - x_target) ** 2 + (y_coords - y_target) ** 2)
    min_distance_idx = np.argmin(distances)

    if distances[min_distance_idx] <= tolerance:
        return scatter.get_facecolors()[min_distance_idx]
    return None


# Пример использования
target_x, target_y = 2, 4
point_color = get_point_color(target_x, target_y, x, y)

print(f"Целевые координаты: ({target_x}, {target_y})")
if point_color is not None:
    print(f"Найденный цвет: {point_color}")
else:
    print("Точка не найдена в пределах допустимой погрешности")
plt.show()