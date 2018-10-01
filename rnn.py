#!/usr/bin/env python3

import copy, numpy as np
import sequences

sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_deriv = lambda x: x * (1 - x)

class SimpleRNN():
    """
    A simple 3-layer RNN.

    Original code from https://youtu.be/cdLUzrjnlr4.
    """

    def __init__(self, alpha, input_dim, hidden_dim, output_dim):
        """
        TODO
        """
        self.alpha = alpha
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights.
        self.synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
        self.synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
        self.synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

    def train(self, train_input, train_output, steps):
        """
        Trains the RNN against training data.
        """
        
        binary_dim = 8

        synapse_0_update = np.zeros_like(self.synapse_0)
        synapse_1_update = np.zeros_like(self.synapse_1)
        synapse_h_update = np.zeros_like(self.synapse_h)

        for i in range(steps):
            # Select random data vectors.
            j = np.random.randint(len(train_input))
            a = train_input[j][0]
            b = train_input[j][1]
            c = train_output[j][0]

            # Network's guess.
            d = np.zeros_like(c)

            overall_error = 0

            layer_2_deltas = []
            layer_1_values = []
            layer_1_values.append(np.zeros(self.hidden_dim))

            for j in range(binary_dim):
                X = np.array([[a[binary_dim - j - 1], b[binary_dim - j - 1]]])
                y = np.array([[c[binary_dim - j - 1]]]).T

                # Hidden layer including influence from previous hidden layer.
                layer_1 = sigmoid(np.dot(X, self.synapse_0) + np.dot(layer_1_values[-1], self.synapse_h))

                # Output layer.
                layer_2 = sigmoid(np.dot(layer_1, self.synapse_1))

                # Calculate errors.
                layer_2_error = y - layer_2
                layer_2_deltas.append(layer_2_error * sigmoid_deriv(layer_2))
                overall_error += np.abs(layer_2_error[0])

                # Round output to nearest integer.
                d[binary_dim - j - 1] = np.round(layer_2[0][0])

                # Store hidden layer for next training iteration.
                layer_1_values.append(copy.deepcopy(layer_1))

            future_layer_1_delta = np.zeros(self.hidden_dim)

            for j in range(binary_dim):
                X = np.array([[a[j], b[j]]])
                layer_1 = layer_1_values[-j - 1]
                prev_layer_1 = layer_1_values[-j - 2]

                # Error at output layer.
                layer_2_delta = layer_2_deltas[-j - 1]

                # Error at hidden layer.
                layer_1_delta = (future_layer_1_delta.dot(self.synapse_h.T) + layer_2_delta.dot(self.synapse_1.T)) * sigmoid_deriv(layer_1)

                synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
                synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
                synapse_0_update += X.T.dot(layer_1_delta)

                future_layer_1_delta = layer_1_delta

            # Update weights
            self.synapse_0 += synapse_0_update * self.alpha
            self.synapse_1 += synapse_1_update * self.alpha
            self.synapse_h += synapse_h_update * self.alpha

            synapse_0_update *= 0
            synapse_1_update *= 0
            synapse_h_update *= 0

            if i % 1000 == 0:
                print("Error: {}".format(overall_error))
                print("Output: {}".format(d))
                print("Expected: {}".format(c))
                out = 0
                for i, x in enumerate(reversed(d)):
                    out += x * 2**i
                #print("{} + {} = {}".format(a_int, b_int, out))
                print()

    def eval(self):
        """
        Evaluates model again a dataset.
        """
        pass
        

def main():
    int2binary = {}
    binary_dim = 8

    largest_number = 2**binary_dim
    binary = np.unpackbits(
        np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
    for i in range(largest_number):
        int2binary[i] = binary[i]

    inputs = []
    outputs = []

    for a in range(int(largest_number / 2)):
        for b in range(int(largest_number / 2)):
            inputs.append((int2binary[a], int2binary[b]))
            outputs.append((int2binary[a + b], ))

    model = SimpleRNN(0.1, 2, 16, 1)
    model.train(inputs, outputs, 30000)

if __name__ == "__main__":
    main()
