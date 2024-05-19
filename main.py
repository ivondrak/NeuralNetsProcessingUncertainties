# This is a sample Python script.
from backpropagation import BackPropagation


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

training_set = [
    ([0.0, 0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ([1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
    ([0.5, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0]),
    ([0.0, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ([1.0, 0.5, 1.0, 0.5, 0.0], [0.0, 1.0, 0.0]),
    ([1.0, 1.0, 0.0, 0.5, 0.5], [1.0, 0.0, 0.0])
 ]

net_input = [0.5, 1.0, 0.0, 1.0, 1.0]


def run_backpropagation():

    # Use a breakpoint in the code line below to debug your script.
    bpnn = BackPropagation(training_set, [5, 10, 10, 3], [0.3, 0.1, 0.1], 1000)
    bpnn.backpropagation()
    output = bpnn.run(net_input)
    print("Result is: ", output)
    print("Mean squared error is: ", bpnn.calculate_mean_squared_error())
    print("Max error of a single neuron is: ", bpnn.calculate_max_error())
    


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_backpropagation()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
