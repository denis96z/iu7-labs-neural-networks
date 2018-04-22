from matrix import Matrix


class Perceptron:
    class PerceptronLayer:
        def __init__(self, perceptron, num_inputs, num_outputs, threshold):
            self.__perceptron__ = perceptron
            self.__weights__ = Matrix(num_outputs, num_inputs)
            self.__weights__.init_random()
            self.__offset__ = Matrix.initialized(num_outputs, 1, threshold)

        def get_num_inputs(self):
            return self.__weights__.get_num_cols()

        def get_num_outputs(self):
            return self.__weights__.get_num_rows()

        def predict(self, in_vector):
            v = (self.__weights__ * in_vector) + self.__offset__
            y = v.apply_func(self.__perceptron__.__act_func__)
            return y

        def learn(self):
            raise NotImplementedError()

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

    def add_layer(self, num_outputs, num_inputs=-1, threshold=0):
        n = len(self.__layers__)
        assert num_inputs != -1 or n > 0
        if len(self.__layers__) > 0:
            num_inputs = self.__layers__[n - 1].get_num_outputs()
        self.__layers__.append(self.__create_layer__(num_inputs, num_outputs, threshold))

    def predict(self, in_vector):
        out_vector = Matrix.from_list(in_vector)
        for layer in self.__layers__:
            out_vector = layer.predict(out_vector)
        return out_vector

    def __create_layer__(self, num_inputs, num_outputs, threshold):
        return Perceptron.PerceptronLayer(self, num_inputs, num_outputs, threshold)
