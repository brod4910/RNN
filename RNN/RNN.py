import numpy as np
import os as os

# load data
data = open("data/kafka.txt", "r").read()

# get the list of chars from the data
chars = list(set(data))

# get the number of chars in the data and also the unique number of chars
data_size, vocab_size = len(data), len(chars)
print("data has %d chars and %d unique chars" % (data_size, vocab_size))

# encode and decode integers to chars and chars to integers

char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

print(char_to_int)
print(int_to_char)

# define hyper-parameters

hidden_size = 100
seq_length = 25
learning_rate = 1e-1

# model parameters

Weights_input_hidden = np.random.rand(hidden_size, vocab_size) * 0.01  # input to hidden state
Weights_hidden_hidden = np.random.rand(hidden_size, hidden_size) * 0.01
Weights_hidden_output = np.random.rand(vocab_size, hidden_size) * 0.01
bias_hidden = np.zeros((hidden_size, 1))
bias_output = np.zeros((vocab_size, 1))

def lossfunction(inputs, targets, hprev):

    input_chars, hidden_output, target_char, normalized_prob = {}, {}, {}, {}

    hidden_output[-1] = np.copy(hprev)
    loss = 0

    #forward pass
    for t in range(len(inputs)):
        input_chars[t] = np.zeros((vocab_size), 1)
        input_chars[t][inputs[t]] = 1
        hidden_output[t] = np.tanh(np.dot(Weights_input_hidden,input_chars[t]) + np.dot(Weights_hidden_hidden, hidden_output[t-1]) + bias_hidden)
        target_char[t] = np.dot(Weights_hidden_output, hidden_output[t]) + bias_output
        normalized_prob[t] = np.exp(target_char[t])/np.sum(np.exp(target_char[t]))
        loss += -np.log(normalized_prob[t][targets[t]], 0)

    # backward pass: compute gradients going backwards
    # intialize vectors for gradient values for each set of weights
    dWxh, dWhh, dWhy = np.zeros_like(Weights_input_hidden), np.zeros_like(Weights_hidden_hidden), np.zeros_like(Weights_hidden_output)
    dbh , dby = np.zeros_like(bias_hidden), np.zeros_like(bias_output)
    dhnext = np.zeros_like(hidden_output[0])

    for t in reversed(range(len(inputs))):
        # output probabilities
        dy = np.copy(normalized_prob[t])
        # derive our first gradient
        dy[targets[t]] -= 1 # backprop into y
        # compute output gradient -  output times hidden states transpose
        # When we apply the transpose weight matrix,
        # we can think intuitively of this as moving the error backward
        # through the network, giving us some sort of measure of the error
        # at the output of the lth layer.
        # output gradient
        dWhy = np.dot(dy, hidden_output[t].T)
        # derivative of output bias
        dby += dy
        # backpropagate
        dh = np.dot(Weights_hidden_output.T, dy) + dhnext # backprop into h

        dhraw = (1 - hidden_output[t] * hidden_output[t]) * dh # backprop through tanh nonlinearity

        dbh += dhraw # derivative of hidden bias

        dWxh += np.dot(dhraw, input_chars[t].T) # derivative of input to hidden state
        dWhh += np.dot(dhraw, hidden_output[t-1].T) # derivative of hidden to hidden
        dhnext = np.dot(Weights_hidden_hidden.T, dhraw)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

    return loss, dWxh, dWhh, dWhy, dbh, dby, hidden_output[len(inputs)-1]

def sample(h, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    n is how many characters to predict
    """

    # create vector
    x = np.zeros((vocab_size, 1))
    # customize it for our seed chars
    x[seed_ix] = 1
    # list to store generated chars
    ixes = []
    # for as many chars as we want to generate
    for t in range(n):
        # a hidden state at a given time step is a function
        # of the input at the same time step modified by a weight matrix
        # added to the hidden state of the previous time step
        # multiplied by its own hidden state to hidden state matrix.
        h = np.tanh(np.dot(Weights_input_hidden, x) + np.dot(Weights_hidden_hidden, h) + bias_hidden)
        # compute output (unnormalized)
        y = np.dot(Weights_hidden_output, h) + bias_output
        # probabilities for next chars
        p = np.exp(y) / np.sum(np.exp(y))
        # pick one with the highest probability
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        # create vector
        x = np.zeros((vocab_size, 1))
        # customize it for the predicted char
        x[ix] = 1
        # add it to the list
        ixes.append(ix)
        txt = ''.join(int_to_char[ix] for ix in ixes)
        print('----\n %s \n----' %(txt, ))


hprev = np.zeros((hidden_size, 1)) # reset RNN memory

sample(hprev, char_to_int['a'], 200)