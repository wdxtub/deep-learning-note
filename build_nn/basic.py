import nn as nn
import numpy as np
import datetime

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

n = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

for i in range(2):
    print("=================#%d Round=====================" % (i+1))
    train_name = "mnist_train_100.csv"
    test_name = "mnist_test_10.csv"
    if i == 1:
        train_name = "mnist_train.csv"
        test_name = "mnist_test.csv"

    print("loading training data", datetime.datetime.now())
    training_data_file = open("data/%s" % train_name, "r")
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    print("loading testing data", datetime.datetime.now())
    testing_data_file = open("data/%s" % test_name, "r")
    testing_data_list = testing_data_file.readlines()
    testing_data_file.close()

    print("start training", datetime.datetime.now())
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

    print("start testing", datetime.datetime.now())
    score_card = []

    for record in testing_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        print(correct_label, "correct label")

        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)

        label = np.argmax(outputs)
        print(label, "network answer")

        if label == correct_label:
            score_card.append(1)
        else:
            score_card.append(0)

    score_card_array = np.asarray(score_card)
    print("performance = ", score_card_array.sum() / score_card_array.size)
