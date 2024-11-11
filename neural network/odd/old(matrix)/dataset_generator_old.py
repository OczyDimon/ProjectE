import numpy as np


rng = np.random.default_rng()

max_size = 8  # Максимальный размер матрицы
number_of_samples = 1000  # Количество сгенерированных матриц для каждой размерности
input_data = []  # Список сгенерированных матриц с их детерминантами
output_data = []  # Список определителей сгенерированных матриц

# for size in range(1, max_size + 1):
#     for _ in range(number_of_samples):
#         # Создание матрицы размерности size*size с рандомными элементами
#         m = rng.integers(low=-100, high=100, size=(size, size))
#         # "Дополнение" матрицы до размерности max_size*max_size единицами по диагонали
#         data = np.block([
#             [m, np.zeros((size, max_size - size))],
#             [np.zeros((max_size - size, size)), np.eye(max_size - size, max_size - size)]
#         ])
#         # Поиск определителя матрицы и округление до 4 знаков после запятой
#         answer = np.round(np.linalg.det(data), 4)
#         input_data.append(data)
#         output_data.append(answer)
#     print(f'Все матрицы размерности {size}/{max_size} сгенерированы')
#
# np.save('input_data.npy', np.array(input_data))
# np.save('output_data.npy', np.array(output_data))

'''
TODO: Сохранение в файл - np.save('my_array.npy', input_data)
Но для начала нагенерировать для размерностей до 10???, каждой по 10???
'''
