from math import exp
from neuralnetworks.perceptron import Perceptron


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_dif(x):
    temp = exp(-x)
    return temp / ((1 + temp) ** 2)


def main():
    p = Perceptron()
    p.set_learning_rate(0.1)
    p.set_act_func(sigmoid, sigmoid_dif)
    p.add_layer(num_inputs=5, num_outputs=20, threshold=0.5)
    p.add_layer(num_outputs=10, threshold=0.3)
    p.add_layer(num_outputs=3, threshold=0.1)
    in_vector, out_vector = [1, 0, 0, 1, 1], [1, 0, 0]
    print("INPUT:")
    print(in_vector)
    print("EXPECTED OUTPUT:")
    print(out_vector)
    print("PREDICTION BEFORE LEARNING:")
    print(p.predict(in_vector))
    print("LEARNING...")
    for i in range(0, 1000):
        p.learn(in_vector, out_vector)
    print("PREDICTION AFTER LEARNING:")
    print(p.predict(in_vector))


if __name__ == "__main__":
    main()
