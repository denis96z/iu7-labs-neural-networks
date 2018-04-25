import numpy as np
from PIL import Image
from keras import Sequential, losses
from keras.layers import Dense
from keras.optimizers import SGD

IMG_SIZE = (10, 10)
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

NUM_INPUTS = IMG_SIZE[0] * IMG_SIZE[1] * 4
NUM_OUTPUTS = len(LETTERS)

NUM_TRAINING_SETS = 100
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.2
TRAINING_SETS_DIR = './../lab01/data/training_sets'

NUM_TEST_SETS = 50
TEST_SETS_DIR = './../lab01/data/test_sets'


def main():
    try:
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
                            batch_size=1, epochs=NUM_EPOCHS,
                            verbose=1, validation_split=VALIDATION_SPLIT)
        score = model.evaluate(x_test, y_test, verbose=1)

        print('Test score: ', score[0])
        print('Test accuracy: ', score[1])
    except:
        print('Error occurred...')


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
    return np.array(x), np.array(y)


if __name__ == "__main__":
    main()
