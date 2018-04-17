from matrix import Matrix


class Perceptron:
    def __init__(self):
        self.__layers__ = []
        self.__act_func__ = lambda x: x
        self.__act_func_dif__ = lambda x: 1
        self.__learning_rate__ = 0.1

    def set_act_func(self, f, f_dif):
        self.__act_func__ = f
        self.__act_func_dif__ = f_dif

    def set_learning_rate(self, rate):
        self.__learning_rate__ = rate

    def add_layer(self, num_outputs, num_inputs=-1):
        n = len(self.__layers__)
        assert num_inputs != -1 or n > 0
        if len(self.__layers__) == 0:
            self.__layers__.append(Matrix(num_outputs, num_inputs))
        else:
            self.__layers__.append(Matrix(num_outputs, self.__layers__[n - 1].get_num_rows()))
        self.__layers__[n].init_random()

    def predict(self, in_vector):
        out_vector = Matrix.from_list(in_vector)
        for i in range(0, len(self.__layers__)):
            out_vector = self.__layers__[i] * out_vector
            out_vector = [self.__act_func__(x) for x in out_vector]
        return out_vector

    def learn(self, in_vector, expected_out_vector):
        n = len(self.__layers__)
        assert n > 0 and len(in_vector) == self.__layers__[0].get_num_cols() and \
            len(expected_out_vector) == self.__layers__[n - 1].get_num_rows()
        v, y = [], []
        v_cur = Matrix.from_list(in_vector)
        for i in range(0, n):
            v_cur = self.__layers__[i] * v_cur
            y_cur = Matrix.from_list([self.__act_func__(x[0]) for x in v_cur])
            v.append(v_cur)
            y.append(y_cur)
            print(self.__layers__[i])
        e__cur = Matrix.from_list(expected_out_vector) - y[n - 1]
        # TODO