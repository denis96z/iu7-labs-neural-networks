from math import exp
from PIL import Image
from neuralnetworks.perceptron import Perceptron
from neuralnetworks.matrix import Matrix

IMG_SIZE = (10, 10)
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

NUM_INPUTS = IMG_SIZE[0] * IMG_SIZE[1] * 4
NUM_OUTPUTS = len(LETTERS)

NUM_TRAINING_SETS = 100
NUM_EPOCHS = 100
TRAINING_SETS_DIR = './data/training_sets'

NUM_TEST_SETS = 10
TEST_SETS_DIR = './data/test_sets'


def main():
    p = Perceptron()

    p.set_learning_rate(0.5)
    p.set_act_func(sigmoid, sigmoid_dif)

    p.add_layer(num_inputs=NUM_INPUTS, num_outputs=256, threshold=0.5)
    p.add_layer(num_outputs=64, threshold=0.1)
    p.add_layer(num_outputs=NUM_OUTPUTS)

    print('LOADING DATA...')
    x_train, y_train = load_sets(TRAINING_SETS_DIR, NUM_TRAINING_SETS)
    x_test, y_test = load_sets(TEST_SETS_DIR, NUM_TEST_SETS)

    print('LEARNING...')
    for i in range(NUM_EPOCHS):
        print('EPOCH #' + str(i))
        max_mse, max_mse_index = 0, 0
        for j in range(NUM_TRAINING_SETS):
            cur_mse = p.learn(x_train[j], y_train[j])
            print('MSE[' + str(j) + ']=' + str(cur_mse))
            if cur_mse > max_mse:
                max_mse, max_mse_index = cur_mse, j
        print('MAX(MSE)=' + str(max_mse) + ' INDEX=' + str(max_mse_index))
        cur_mse = p.learn(x_train[max_mse_index], y_train[max_mse_index])
        while (cur_mse < max_mse) and (cur_mse > 0.01):
            max_mse = cur_mse
            print('MSE[' + str(max_mse_index) + ']=' + str(cur_mse))
            cur_mse = p.learn(x_train[max_mse_index], y_train[max_mse_index])

    print('TESTING...')
    for i in range(len(x_test)):
        print('EXPECTED:')
        print(y_test[i])
        print('RESULT:')
        y = p.predict(x_test[i]).to_list()
        index = 0
        for i in range(1, len(y)):
            if y[i] > y[index]:
                index = i
        print('INDEX OF MAX=' + str(index))
        print(y)


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_dif(x):
    temp = exp(-x)
    return temp / ((1 + temp) ** 2)


def make_file_path(root, set_num, letter):
    return root + '/' + str(set_num) + '/' + letter + '.png'


def make_expected_output_vector(letter):
    index = ord(letter) - ord('A')
    vector = [0 for i in LETTERS]
    vector[index] = 1
    return vector


def load_image_from_file(path):
    img = Image.open(path).resize(IMG_SIZE)
    bts = [(0 if b == 0 else 1) for b in img.tobytes()]
    return bts


def load_sets(root_dir, num_sets):
    x, y = [], []
    for cur_set in range(num_sets):
        for letter in LETTERS:
            path = make_file_path(root_dir, cur_set, letter)
            cur_x = load_image_from_file(path)
            cur_y = make_expected_output_vector(letter)
            x.append(cur_x), y.append(cur_y)
    return x, y


if __name__ == "__main__":
    main()
