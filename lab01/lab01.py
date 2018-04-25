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

    p.set_learning_rate(0.1)
    p.set_act_func(sigmoid, sigmoid_dif)

    p.add_layer(num_inputs=NUM_INPUTS, num_outputs=20, threshold=0.5)
    p.add_layer(num_outputs=NUM_OUTPUTS, threshold=0.1)

    print('LEARNING...')
    for i in range(NUM_EPOCHS):
        for tr_set in range(NUM_TRAINING_SETS):
            for letter in LETTERS:
                path = make_file_path(TRAINING_SETS_DIR, tr_set, letter)
                in_vector = load_image_from_file(path)
                expected_out_vector = make_expected_output_vector(letter)
                p.learn(in_vector, expected_out_vector)
                print('LEARNED "' + letter + '" FROM SET #' + str(tr_set))
        print('EPOCH #' + str(i) + ' passed')

    print('TESTING...')
    for ts_set in range(NUM_TEST_SETS):
        for letter in LETTERS:
            path = make_file_path(TEST_SETS_DIR, ts_set, letter)
            in_vector = load_image_from_file(path)
            predicted_index = p.predict(in_vector)
            print('SET: #' + str(ts_set) + '; EXPECTED: "' + letter +
                  '"; PREDICTED: "' + LETTERS[predicted_index] + '"')


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_dif(x):
    temp = exp(-x)
    return temp / ((1 + temp) ** 2)


def make_file_path(root, set, letter):
    return root + '/' + str(set) + '/' + letter + '.png'


def make_expected_output_vector(letter):
    index = ord(letter) - ord('A')
    vector = [0 for i in range(len(LETTERS))]
    vector[index] = 1
    return vector


def load_image_from_file(path):
    img = Image.open(path).resize(IMG_SIZE)
    bts = [(0 if b == 0 else 1) for b in img.tobytes()]
    return bts


if __name__ == "__main__":
    main()
