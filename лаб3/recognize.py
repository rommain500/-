

import numpy as np


#функция поиска максимального веса
def search_max_in(img):
    resp = list(range(0, 10))
    for i in range(0, 10):
        r = w[i] * img
        r = np.maximum(np.sum(r) + b[i], 0)  # relu
        resp[i] = r

    return np.argmax(resp)

#получаем данные от формочки
def get_array():
    import pyperclip
    return pyperclip.paste()


#подгружаем коэффициенты
w = np.load('w.npy', mmap_mode='r')
b = np.load("b.npy", mmap_mode='r')

print("Программа распознования:")
while True:
    print('--------------------')
    print("Для распознавания нажимайте enter")
    print("(Введите 'no' для завершения работы)")
    inp = input()
    if inp == "no":
        break
    a = get_array()
    a = a.split(',')
    b = []
    for i in a:
        b.append(int(i))

    print('Скорее всего это:', search_max_in(b))
