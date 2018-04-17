from matrix import Matrix


class Perceptron:
    def __init__(self):
        self.__layers__ = []
        self.__act_func__ = lambda x: x
        self.__learning_rate__ = 0.1

    def set_act_func(self, f):
        self.__act_func__ = f

    def set_learning_rate(self, rate):
        self.__learning_rate__ = rate

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
