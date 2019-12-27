##########################################################################
 ######### inspired by deeplearning.ai dinosaur island exercise #########
##########################################################################
import numpy as np
from matplotlib import pyplot as plt

with open("dinos.txt") as f:
    dinos = f.readlines()
dinos = [x.lower().strip() for x in dinos]
chars = sorted(list(set(open('dinos.txt', 'r').read().lower())))

vocab_size = len(chars)
char_to_idx = {char:idx for idx, char in enumerate(chars)}
idx_to_char = {idx:char for idx, char in enumerate(chars)}
print(f'characters: {chars}')
print(f'number of dino names: {len(dinos)}')


class RNN:
    parameters = {}
    
    def __init__(self, hidden_size, vocab_size, words, learning_rate=0.01):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.parameters['Wxh'] = np.random.randn(hidden_size, vocab_size) * 0.01
        self.parameters['Whh'] = np.random.randn(hidden_size, hidden_size) * 0.01
        self.parameters['Why'] = np.random.randn(vocab_size, hidden_size) * 0.01
        self.parameters['bh'] = np.random.randn(hidden_size, 1)
        self.parameters['by'] = np.random.randn(vocab_size, 1)
        self.words = words

    def forward_prop(self, x, y, h_prev):
        h, y_hat = {}, []
        loss = 0
        h[-1] = h_prev
        for t in range(len(x)):            
            h[t] = np.tanh(self.parameters['Wxh']@x[t] + self.parameters['Whh']@h[t-1] + self.parameters['bh'])
            y_hat.append(self.softmax(self.parameters['Why']@h[t] + self.parameters['by']))
            loss -= self.loss_function(y_hat[t], y[t])
        
        cache = (x, h, y_hat)
        return loss, cache

    #https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy, http://lyy1994.github.io/machine-learning/2016/05/11/softmax-cross-entropy-derivative.html
    def back_prop(self, cache, y, gradients): 
        (x, h, y_hat) = cache
        gradients['dWxh'] = np.zeros(self.parameters['Wxh'].shape)
        gradients['dWhh'] = np.zeros(self.parameters['Whh'].shape)
        gradients['dbh'] = np.zeros(self.parameters['bh'].shape)
        gradients['dWhy'] = np.zeros(self.parameters['Why'].shape)
        gradients['dby'] = np.zeros(self.parameters['by'].shape)
        gradients['dhprev'] = np.zeros(h[0].shape)
        steps = len(x)
        
        for t in reversed(range(steps)):
            dy_hat = y_hat[t] - y[t]
            gradients['dWhy'] += dy_hat@h[t].T
            gradients['dby'] += dy_hat
            dh = self.parameters['Why'].T@dy_hat + gradients['dhprev']
            tanraw = (1 - h[t] * h[t]) * dh
            gradients['dbh'] += tanraw
            gradients['dWhh'] += tanraw @ h[t-1].T
            gradients['dWxh'] += tanraw @ x[t].T
            gradients['dhprev'] = self.parameters['Whh'].T @ tanraw
        return gradients

    def clip_gradients(self, gradients, clip_val = 1):
        for g in [gradients['dWhy'], gradients['dbh'], gradients['dby'], gradients['dWhh'], gradients['dWxh'], gradients['dhprev']]:
            np.clip(g, -clip_val, clip_val, out=g)
        return gradients

    def update_params(self, gradients):
        self.parameters['Wxh'] -= self.learning_rate * gradients['dWxh']
        self.parameters['Whh'] -= self.learning_rate * gradients['dWhh']
        self.parameters['Why'] -= self.learning_rate * gradients['dWhy']
        self.parameters['by'] -= self.learning_rate * gradients['dby']
        self.parameters['bh'] -= self.learning_rate * gradients['dbh']

    def train(self, iterations=50000):
        np.random.shuffle(self.words)
        a_prev =  np.zeros((self.hidden_size, 1))
        loss = 0
        history = []
        for iteration in range(iterations):
            word = self.words[iteration % len(self.words)]
            idx_word = [self.one_hot(char_to_idx[z], self.vocab_size) for z in word]
            x = [np.zeros((self.vocab_size, 1))] + idx_word
            y = idx_word + [self.one_hot(char_to_idx['\n'], self.vocab_size)]
            current_loss, cache = self.forward_prop(x, y, a_prev)
            loss = self.smooth_loss(loss, current_loss)
            gradients = {}
            gradients = self.back_prop(cache, y, gradients)
            gradients = self.clip_gradients(gradients)
            self.update_params(gradients)

            if (iteration % 2000) == 0:
                print(f'\niteration: {iteration} loss: {loss}') 
                for _ in range(5):
                    indices = self.sample()
                    vec = ''
                    for z in indices[:-1]:
                        vec += idx_to_char[z]
                    print(f'sampled name: {vec}')
            
            if (iteration % 1000) == 0:
                history.append(loss)
        self.plot_history(history)

    def one_hot(self, idx, v_size):
        vec = np.zeros((v_size, 1))
        vec[idx] = 1
        return vec

    def softmax(self, z):
        z_u = np.exp(z)
        return z_u / np.sum(z_u)

    def loss_function(self, y_hat, y): #categorical cross entropy
        #return np.log(np.sum(np.multiply(y_hat, y)))
        return np.log(y_hat[y.argmax()])

    def smooth_loss(self, loss, current_loss):
        return current_loss * 0.01 + loss * 0.99

    def sample(self):
        x = np.zeros((self.vocab_size, 1))
        h_prev = np.zeros((self.parameters['Whh'].shape[1], 1))
        indices = []
        counter = 0
        idx = -1
        newline_character = char_to_idx['\n']
    
        while (idx != newline_character and counter != 50):
            h = np.tanh(self.parameters['Wxh']@x + self.parameters['Whh']@h_prev + self.parameters['bh'])
            h_prev = h
            y_hat = self.softmax(self.parameters['Why']@h + self.parameters['by'])
            idx = np.random.choice(np.arange(0, self.vocab_size, 1), p=np.ravel(y_hat))
            indices.append(idx)
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            h_prev = h
            counter += 1
        
        if counter == 50:
            indices.append(char_to_idx['\n'])
        
        return indices
    
    def plot_history(self, history):
        plt.plot(np.arange(1000, len(history)*1000, 1000), history[1:])
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()


rnn = RNN(50, vocab_size, dinos)
rnn.train()