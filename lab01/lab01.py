from math import exp
from PIL import Image
from neuralnetworks.perceptron import Perceptron


IMG_SIZE = (5, 8)
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

NUM_INPUTS = IMG_SIZE[0] * IMG_SIZE[1]
NUM_OUTPUTS = len(LETTERS)


NUM_TRAINING_SETS = 4
NUM_REPEATS = 1000
TRAINING_SETS_DIR = './data/training_sets'

NUM_TEST_SETS = 1
TEST_SETS_DIR = './data/test_sets'


def main():
    p = Perceptron()

    p.set_learning_rate(0.1)
    p.set_act_func(sigmoid, sigmoid_dif)

    p.add_layer(num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS, threshold=0.5)

    print('LEARNING...')
    for set in range(1, NUM_TRAINING_SETS + 1):
        for letter in LETTERS:
            path = make_file_path(TRAINING_SETS_DIR, set, letter)
            in_vector = load_image_from_file(path)
            expected_out_vector = make_expected_output_vector(letter)
            for i in range(NUM_REPEATS):
                p.learn(in_vector, expected_out_vector)
            print('LEARNED "' + letter + '" FROM SET #' + str(set))

    print('TESTING...')
    for set in range(1, NUM_TEST_SETS + 1):
        for letter in LETTERS:
            path = make_file_path(TEST_SETS_DIR, set, letter)
            in_vector = load_image_from_file(path)
            predicted_index = p.predict(in_vector)
            print('SET: #' + str(set) + '; EXPECTED: "' + letter +
                  '"; PREDICTED: "' + LETTERS[predicted_index] + '"')


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_dif(x):
    temp = exp(-x)
    return temp / ((1 + temp) ** 2)


def make_file_path(root, set, letter):
    return root + '/' + str(set) + '/' + letter + '.bmp'


def make_expected_output_vector(letter):
    index = ord(letter) - ord('A')
    vector = [0 for i in range(len(LETTERS))]
    vector[index] = 1
    return vector


def load_image_from_file(path):
    try:
        img = Image.open(path).resize(IMG_SIZE)
        return [0 if px == 255 else 1 for px in img.tobytes()]
    except IOError:
        return None


if __name__ == "__main__":
    main()
