import keras
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


learning_rate = 0.001
n_epochs = 20
batch_size = 100
n_batches = int(mnist.train.num_examples/batch_size)
# number of pixels in the MNIST image as number of inputs
n_inputs = 784
n_outputs = n_inputs
# number of hidden layers
n_layers = 2
# neurons in each hidden layer
n_neurons = [512,256]
# add decoder layers:
n_neurons.extend(list(reversed(n_neurons)))
n_layers = n_layers * 2

model = Sequential()

# add input to first layer
model.add(Dense(units=n_neurons[0], activation='relu', 
    input_shape=(n_inputs,)))

for i in range(1,n_layers):
    model.add(Dense(units=n_neurons[i], activation='relu'))

# add last layer as output layer
model.add(Dense(units=n_outputs, activation='linear'))
 
model.summary()

model.compile(loss='mse',
    optimizer=keras.optimizers.Adam(lr=learning_rate),
    metrics=['accuracy'])

model.fit(mnist.train.images, mnist.train.images,batch_size=batch_size,
    epochs=n_epochs)


