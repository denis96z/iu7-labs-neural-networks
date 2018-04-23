import numpy as np
from PIL import Image
from keras import Sequential, losses
from keras.layers import Dense
from keras.optimizers import SGD

IMG_SIZE = (5, 8)
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

NUM_INPUTS = IMG_SIZE[0] * IMG_SIZE[1]
NUM_OUTPUTS = len(LETTERS)


NUM_TRAINING_SETS = 4
NUM_REPEATS = 1000
TRAINING_SETS_DIR = './../lab01/data/training_sets'

NUM_TEST_SETS = 1
TEST_SETS_DIR = './../lab01/data/test_sets'


def main():
    x_train, y_train = load_sets(TRAINING_SETS_DIR, NUM_TRAINING_SETS)
    x_train = x_train.astype('float32') / 255
    y_train = y_train.astype('float32') / 255

    x_test, y_test = load_sets(TEST_SETS_DIR, NUM_TEST_SETS)
    x_test = x_test.astype('float32') / 255
    y_test = y_test.astype('float32') / 255

    model = Sequential()
    model.add(Dense(NUM_OUTPUTS, input_shape=(NUM_INPUTS,),
                    kernel_initializer='random_normal',
                    use_bias=True,
                    bias_initializer='random_uniform',
                    activation='sigmoid'))
    model.summary()
    model.compile(loss=losses.mean_squared_error,
                  optimizer=SGD(), metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=1, epochs=1000,
                        verbose=1, validation_split=0)
    score = model.evaluate(x_test, y_test, verbose=1)

    print('Test score: ', score[0])
    print('Test accuracy: ', score[1])


def make_file_path(root, set, letter):
    return root + '/' + str(set) + '/' + letter + '.bmp'


def make_expected_output_vector(letter):
    index = ord(letter) - ord('A')
    vector = [0 for i in LETTERS]
    vector[index] = 1
    return vector


def load_image_from_file(path):
    try:
        img = Image.open(path).resize(IMG_SIZE)
        return [0 if px == 255 else 1 for px in img.tobytes()]
    except IOError:
        return None


def load_sets(root_dir, num_sets):
    x, y = [], []
    for cur_set in range(1, num_sets + 1):
        for letter in LETTERS:
            path = make_file_path(root_dir, cur_set, letter)
            cur_x = load_image_from_file(path)
            cur_y = make_expected_output_vector(letter)
            x.append(cur_x), y.append(cur_y)
    return np.array(x), np.array(y)


if __name__ == "__main__":
    main()