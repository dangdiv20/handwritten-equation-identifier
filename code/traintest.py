from __future__ import absolute_import
from matplotlib import pyplot as plt
from numpy.lib.function_base import _DIMENSION_NAME, select
from tensorflow.python.framework.tensor_conversion_registry import get
from tensorflow.python.ops.gen_math_ops import exp
from tensorflow.python.ops.gen_nn_ops import MaxPool
import os
import tensorflow as tf
import numpy as np
import random
import math
from preprocess import get_data
from model import Model
from one_hot import decode, encode
from extract import Extractor

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. 
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training) 
    :param train_labels: train labels (all labels to use for training)
    :return: loss list and accuracy
    '''

    #loop through data in batches
    accuracy = 0
    j = 0
    for i in range(0, train_inputs.shape[0], model.batch_size):

        if(i + model.batch_size > train_inputs.shape[0]):
            break

        #maybe change the shape of the inputs depending on what they are
        batch_input = train_inputs[i : i + model.batch_size]
        batch_label = train_labels[i : i + model.batch_size]

        #calculated logit and loss
        with tf.GradientTape() as tape:
            logits = model.call(batch_input)
            loss = model.loss(logits, batch_label)
            
            accuracy += model.accuracy(logits, batch_label)
            model.loss_list.append(loss)

        #gets the gradient
        gradients = tape.gradient(loss, model.trainable_variables)

        #applies the optimizer
        model.optimization.apply_gradients(zip(gradients, model.trainable_variables))

        j+=1

    return model.loss_list, accuracy/j



def test_characters(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels.

    :param test_inputs: test data (all images to be tested), 
    :param test_labels: test labels (all corresponding labels),
    :return: test accuracy - testing accuracy
    """
    accuracy = 0
    num_batches = 0

    #loop through the data in batches
    for i in range(0, len(test_inputs), model.batch_size):

        if(i + model.batch_size > len(test_inputs)):
            break

        #get the batched inputs and labels
        batch_input = test_inputs[i : i + model.batch_size]
        batch_label = test_labels[i : i + model.batch_size]

        #get logits and calculated accuracy
        logits = model.call(batch_input)
        accuracy += model.accuracy(logits, batch_label)

        num_batches+=1

    return accuracy/num_batches


def test_expressions(model, test_inputs, test_labels, extractor1):
    """
    Tests the model on the test inputs and labels fo full mathematical 
    expressions.
    
    :param test_inputs: test data (all images to be tested), 
    :param test_labels: test labels (all corresponding labels),
    :return: test accuracy - testing accuracy
    """
    accuracy = 0

    #loop through the data in batches

    for i in range(len(test_inputs)):

        # each expression acts as a batch
        expression_input = test_inputs[i]
        expression_input = np.array(expression_input)
        #reshape each expression for testing
        expression_input = np.reshape(expression_input, (-1,32,32,1))
        expression_label = test_labels[i]

        #get logits and calculated accuracy
        logits = model.call(expression_input)

        #create a fucntion that measures accuracy for expressions
        print("Predicted label: ", decode_expression(logits, extractor1))
        print("True Label: ", decode_expression(expression_label, extractor1))
        accuracy += model.accuracy(logits, expression_label)

    return accuracy/len(test_inputs)

def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 


    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def decode_expression(probabilities, extractor1):
    """
    Decodes the expression from probabilities into actual math symbols

    param probabilities: the probabilities returned by the model
    param extractor: the extractor that was used to preprocess the data
    """
    predicted_labels = np.argmax(probabilities, axis=1)
    output_symbols = []
    for i in range(len(predicted_labels)):
        symbol = decode(predicted_labels[i], extractor1.classes)
        output_symbols += symbol
    return output_symbols


def main():

    #get and load data
    train_inputs, train_labels, test_inputs, test_labels, test_char_inputs, test_char_labels, extractor1 = get_data()
    print("Preprocessing Completed!")

    model = Model()

    #train model
    for i in range(1):
        loss_list, accuracy = train(model, train_inputs, train_labels)
        print("Epoch",i , " ", accuracy)
        print("Loss:", tf.reduce_mean(model.loss_list))
    visualize_loss(loss_list)

    print("Accuracy for Training", accuracy)
    
    # test model on characters

    acc_1 = test_characters(model, test_char_inputs, test_char_labels)
     
    print("Accuracy for testing characters: ", acc_1)

    # test model from expression
    acc = test_expressions(model, test_inputs, test_labels, extractor1)

    print("Accuracy for testing expression", acc)
    pass



if __name__ == '__main__':
    main()