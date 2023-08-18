import functions as fn
import numpy as np
import cost as cs
import os
import new_neural_network as nnn

from PIL import Image


def check_image(file_path):
    sizes = [28 * 28, 64, 64, 10]
    n = nnn.NeuralNetwork(sizes)

    n.from_file('network_cex.json')

    image = np.array(Image.open(file_path).convert('L'))
    a = np.transpose([[aa / 255 for aa in image.flatten()]])

    ans = n.calculate(a, fn.sig)

    print([round(ans[iii][0], 2) for iii in range(len(ans))])
    num = 0
    for i in range(len(ans)):
        if (ans[i][0] > ans[num][0]):
            num = i
    print("Це точнo ", num, "!", sep='')


def new_main():
    # sizes = [28 * 28, 128, 64, 16, 10]
    sizes = [28 * 28, 64, 64, 10]
    n = nnn.NeuralNetwork(sizes)

    n.from_file('network_cex.json')
    # your path to 'train' folder
    dir_path = 'C:\\mnist_train\\train'

    right_ans = 0
    all_ans = 0
    iiii = 0

    paths = [os.path.join(dir_path, p) for p in os.listdir(dir_path)]

    while True:
        np.random.shuffle(paths)

        for pth in paths:
            if os.path.isfile(pth):
                image = np.array(Image.open(pth).convert('L'))
                a = np.transpose([[aa / 255 for aa in image.flatten()]])

                digit = int(pth[-5])
                y = np.zeros(shape=(10, 1))
                y[digit] = 1

                # ans = n.calculate(a, fn.ReLU, fn.softmax)
                ans = n.calculate(a, fn.sig)

                cost = cs.cross_entropy(ans, y)

                all_ans += 1
                if max(ans) == ans[digit][0]:
                    right_ans += 1

                iiii += 1

                if iiii % 100 == 0:
                    print([iiii], digit, [cost], right_ans / all_ans * 100,
                          [round(ans[iii][0], 2) for iii in range(len(ans))])
                if iiii % 1000 == 0:
                    right_ans = 0
                    all_ans = 0
                    n.to_file('network_cex.json')
                n.instant_back_propagation(0.01, y, fn.dersig, cs.der_cross_entropy)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    check_image('mynum.png')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
