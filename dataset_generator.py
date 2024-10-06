import numpy as np


rng = np.random.default_rng()

max_size = 10
number_of_samples = 10
input_data = []


for size in range(1, max_size + 1):
    for j in range(number_of_samples):
        m = rng.integers(low=-100, high=100, size=(size, size))
        data = np.block([
            [m, np.zeros((size, max_size - size))],
            [np.zeros((max_size - size, size)), np.eye(max_size - size, max_size - size)]
        ])
        answer = np.round(np.linalg.det(data), 4)
        input_data.append([data, answer])

print(input_data)

'''
TODO: Сохранение в файл - np.save('my_array.npy', input_data)
Но для начала нагенерировать для размерностей до 20???, каждой по 10000???
'''
