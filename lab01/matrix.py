import random


class Matrix:
    def __init__(self, num_rows, num_cols):
        self.__items__ = [[0 for i in range(num_cols)] for j in range(num_rows)]
        self.__num_rows__ = num_rows
        self.__num_cols__ = num_cols

    def init_random(self):
        for i in range(0, self.__num_rows__):
            for j in range(0, self.__num_cols__):
                self.__items__[i][j] = random.random()

    @staticmethod
    def identity(num_rows):
        result = Matrix(num_rows, num_rows)
        for i in range(0, num_rows):
            result[i][i] = 1

    @staticmethod
    def from_list(lst):
        result = Matrix(len(lst), 1)
        result.__items__ = [[x] for x in lst]
        return result

    def get_num_rows(self):
        return self.__num_rows__

    def get_num_cols(self):
        return self.__num_cols__

    def __getitem__(self, index):
        return self.__items__[index]

    def __setitem__(self, index, item):
        self.__items__[index] = item

    def __add__(self, other):
        assert self.__num_rows__ == other.__num_rows__
        assert self.__num_cols__ == other.__num_cols__
        result = Matrix(self.__num_rows__, self.__num_cols__)
        for i in range(0, self.__num_rows__):
            for j in range(0, self.__num_cols__):
                result.__items__[i][j] = self.__items__[i][j] + other.__items__[i][j]
        return result

    def __sub__(self, other):
        assert self.__num_rows__ == other.__num_rows__
        assert self.__num_cols__ == other.__num_cols__
        result = Matrix(self.__num_rows__, self.__num_cols__)
        for i in range(0, self.__num_rows__):
            for j in range(0, self.__num_cols__):
                result.__items__[i][j] = self.__items__[i][j] - other.__items__[i][j]
        return result

    def __mul__(self, other):
        assert self.__num_cols__ == other.__num_rows__
        result = Matrix(self.__num_rows__, other.__num_cols__)
        for i in range(0, self.__num_rows__):
            for j in range(0, other.__num_cols__):
                s = 0
                for k in range(0, self.__num_cols__):
                    s += self.__items__[i][k] * other.__items__[k][j]
                result.__items__[i][j] = s
        return result

    def __iter__(self):
        return self.__items__.__iter__()

    def __str__(self):
        result = ""
        for i in range(0, self.__num_rows__):
            result += str(self.__items__[i]) + "\n"
        return result
