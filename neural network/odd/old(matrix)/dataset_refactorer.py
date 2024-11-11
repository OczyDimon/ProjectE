from PIL import Image
from random import randint

import os

from dataset_generator import max_size, number_of_samples

# pdf_document = 'book.pdf'
#
# with fitz.open(pdf_document) as doc:
#     for num, page_ in enumerate(doc.pages()):
#         page = doc.load_page(num)
#         pix = page.get_pixmap(matrix=fitz.Matrix(2000 / 1000, 2000 / 1000))
#         pix.save(f'book_sheets/page_{num}.png')


# for size_num in range(2, max_size + 1):
#     for sample_num in range(number_of_samples):
#
#         page_num = randint(1, 323)
#
#         img_textbook = Image.open(f'book_sheets/page_{page_num}.png')
#
#         img_opening_bracket = Image.open('(.png')
#         img_closing_bracket = Image.open(').png')
#         matrix_1p = size_num
#         matrix_2p = sample_num
#         img_matrix = Image.open(f'matrixes/matrix_{matrix_1p}_{matrix_2p}.png')
#
#         textbook_width, textbook_height = img_textbook.size
#         matrix_width, matrix_height = img_matrix.size
#
#         img_opening_bracket = img_opening_bracket.resize((50, matrix_height))
#         img_closing_bracket = img_closing_bracket.resize((50, matrix_height))
#
#         opening_bracket_width, opening_bracket_height = img_opening_bracket.size
#         closing_bracket_width, closing_bracket_height = img_closing_bracket.size
#
#         x_opening_bracket = randint(0, matrix_width - opening_bracket_width)
#         y_opening_bracket = randint(0, textbook_height - opening_bracket_height)
#
#         x_matrix = x_opening_bracket + opening_bracket_width
#         y_matrix = y_opening_bracket
#
#         x_closing_bracket = x_matrix + matrix_width
#         y_closing_bracket = y_matrix
#
#         img_textbook.paste(img_matrix, (x_matrix, y_matrix), img_matrix)
#         img_textbook.paste(img_opening_bracket, (x_opening_bracket, y_opening_bracket), img_opening_bracket)
#         img_textbook.paste(img_closing_bracket, (x_closing_bracket, y_closing_bracket), img_closing_bracket)
#
#         img_save = img_textbook.resize((1200, 1600))
#         img_save.save(f'matrixes_noised/page_{page_num}_matrix_{matrix_1p}_{matrix_2p}.png')
#
#         img_textbook.close()
#         img_opening_bracket.close()
#         img_closing_bracket.close()
#         img_matrix.close()
#         img_save.close()
#
#     print(f'Все матрицы размерности {size_num}/{max_size} сгенерированы')


file_names = []
for file_name in os.listdir('matrixes_noised'):
    if os.path.isfile(os.path.join('matrixes_noised', file_name)):
        file_names.append(file_name)

for file_name in file_names:
    file_name = file_name.replace('.png', '')
    file_name_list = file_name.split('_')
    page_num = file_name_list[1]
    matrix_size = file_name_list[3]
    matrix_num = file_name_list[4]
