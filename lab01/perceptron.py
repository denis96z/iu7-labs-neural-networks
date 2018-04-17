from matrix import Matrix


class Perceptron:
    def __init__(self):
        self.__layers__ = []

    def add_layer(self, num_outputs, num_inputs=-1):
        n = len(self.__layers__)
        assert num_inputs != -1 or n > 0
        if len(self.__layers__) == 0:
            self.__layers__.append(Matrix(num_outputs, num_inputs))
        else:
            self.__layers__.append(Matrix(num_outputs, self.__layers__[n - 1].get_num_rows()))

    def predict(self, in_vector):
        out_vector = Matrix(len(in_vector), 1)
        for i in range(0, len(self.__layers__)):
            out_vector = self.__layers__[i] * out_vector
        return out_vector