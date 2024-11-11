import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng()

max_size = 8  # Максимальный размер матрицы
number_of_samples = 1000  # Количество сгенерированных матриц для каждой размерности
output_data_info = []  # Список сгенерированных матриц с их детерминантами
output_data_matrixes = []  # Список определителей сгенерированных матриц


# def save_matrix_image(matrix, filename):
#     plt.figure(figsize=(0.1, 0.1))
#     plt.axis('off')
#
#     text = '\n'.join([' '.join(map(str, row)) for row in matrix])
#     plt.text(0, 1, text, fontsize=24, ha='center', va='center')
#
#     plt.savefig(filename, bbox_inches='tight')
#     plt.close()
#
#
# for size in range(2, max_size + 1):
#     for i in range(number_of_samples):
#         # Создание матрицы размерности size*size с рандомными элементами
#         m = rng.integers(low=-100, high=100, size=(size, size))
#         save_matrix_image(m, f'matrixes/matrix_{size}_{i}.png')
#         data = np.zeros((max_size, max_size))
#         data[:size, :size] = m
#
#         output_data_info.append([size, i])
#         output_data_matrixes.append(data)
#
#     print(f'Все матрицы размерности {size}/{max_size} сгенерированы')


np.save('input_data.npy', np.array(output_data_info))
np.save('output_data.npy', np.array(output_data_matrixes))
