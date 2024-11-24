import os

buffer_len_bot = 20


def update_buffer(queue, img_path):
    try:
        queue.append(img_path)
        if len(queue) > buffer_len_bot:
            g = queue.pop(0)
            os.remove(g)

    except Exception as ex:
        print(ex)
