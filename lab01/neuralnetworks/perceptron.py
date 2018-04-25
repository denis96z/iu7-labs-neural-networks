from neuralnetworks.matrix import Matrix


class Perceptron:
    class PerceptronLayer:
        def __init__(self, perceptron, num_inputs, num_outputs, threshold):
            self.__perceptron__ = perceptron
            self.__weights__ = Matrix(num_outputs, num_inputs)
            self.__weights__.init_random()
            self.__offset__ = Matrix.initialized(num_outputs, 1, threshold)
            self.__latest_x_input__ = None
            self.__latest_v_output__ = None
            self.__latest_y_output__ = None

        def get_num_inputs(self):
            return self.__weights__.get_num_cols()

        def get_num_outputs(self):
            return self.__weights__.get_num_rows()

        def predict(self, in_vector):
            self.__latest_x_input__ = in_vector
            self.__latest_v_output__ = (self.__weights__ * in_vector) + self.__offset__
            self.__latest_y_output__ = self.__latest_v_output__.apply_func(self.__perceptron__.__act_func__)
            return self.__latest_y_output__

        def learn_from_vector(self, expected):
            err = expected - self.__latest_y_output__
            return err, self.learn_from_error(err)

        def learn_from_error(self, err):
            dif = self.__latest_v_output__.apply_func(self.__perceptron__.__act_func_dif__)
            grad = Matrix(dif.get_num_rows(), 1)
            for i in range(0, dif.get_num_rows()):
                grad[i][0] = err[i][0] * dif[i][0]
            n_rows, n_cols = grad.get_num_rows(), self.__latest_x_input__.get_num_rows()
            w_adj = Matrix(n_rows, n_cols)
            for i in range(0, n_rows):
                for j in range(0, n_cols):
                    w_adj[i][j] = self.__perceptron__.__learning_rate__ * grad[i][0] * self.__latest_x_input__[j][0]
            self.__weights__ = self.__weights__ + w_adj
            return (grad.transposed() * self.__weights__).transposed()

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
        assert len(self.__layers__) > 0
        out_vector = Matrix.from_list(in_vector)
        for layer in self.__layers__:
            out_vector = layer.predict(out_vector)
        return out_vector

    def learn(self, in_vector, expected_out_vector):
        self.predict(in_vector)
        n = len(self.__layers__)
        expected_out_vector = Matrix.from_list(expected_out_vector)
        out_err, err = self.__layers__[n - 1].learn_from_vector(expected_out_vector)
        for i in reversed(range(n - 1)):
            err = self.__layers__[i].learn_from_error(err)
        return Perceptron.__count_mse__(out_err)

    def __create_layer__(self, num_inputs, num_outputs, threshold):
        return Perceptron.PerceptronLayer(self, num_inputs, num_outputs, threshold)

    @staticmethod
    def __count_mse__(err_vector):
        err_vector = err_vector.to_list()
        n = len(err_vector)
        s = 0
        for x in err_vector:
            s += x * x
        return s / n

    @staticmethod
    def __find_max_index__(out_vector):
        max_index = 0
        for i in range(1, out_vector.get_num_rows()):
            if out_vector[i][0] > out_vector[max_index][0]:
                max_index = i
        return max_index
