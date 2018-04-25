from math import exp
from PIL import Image
from neuralnetworks.perceptron import Perceptron


IMG_SIZE = (10, 10)
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

NUM_INPUTS = IMG_SIZE[0] * IMG_SIZE[1] * 4
NUM_OUTPUTS = len(LETTERS)


NUM_TRAINING_SETS = 100
NUM_EPOCHS = 100
TRAINING_SETS_DIR = './data/training_sets'

NUM_TEST_SETS = 1
TEST_SETS_DIR = './data/test_sets'


def main():
    p = Perceptron()

    p.set_learning_rate(0.5)
    p.set_act_func(sigmoid, sigmoid_dif)

    p.add_layer(num_inputs=NUM_INPUTS, num_outputs=512, threshold=0.5)
    p.add_layer(num_outputs=256, threshold=0.2)
    p.add_layer(num_outputs=64, threshold=0.1)
    p.add_layer(num_outputs=NUM_OUTPUTS, threshold=0.05)


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
