## Implements the fully connected layer ##
import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        # Parent/Super constructor called
        super().__init__()
        # trainable is inherited from BaseLayer
        self.trainable = True
        # Setting o/p and i/p size for the class
        self.output_size = output_size
        self.input_size = input_size
        # Random initialisation of the weights using Numpy between range 0 to 1
        self.weights = np.random.uniform(0., 1., (self.input_size + 1, self.output_size))
        # Input tensor used in forward pass
        self.input_tensor = None
        # Declaring the getter/setter property optimizer
        self.optimizer = None
        # Gradient weights returns the gradient with respect to the weights, after they have been calculated in the backward-pass.
        self.gradient_weights = None

    def initialize(self , weights_initializers , bias_initializers):

        fan_in = self.input_size
        fan_out = self.output_size
        self.weights[:-1] = weights_initializers.initialize((self.input_size , self.output_size) , fan_in , fan_out)
        self.weights[-1] = bias_initializers.initialize((1 , self.output_size) , fan_in , fan_out)

    def optimizer(self):
        #Getter property for private optimizer attribute
        return self.__optimizer

    def optimizer(self, val):
        #Setter property for private optimizer attribute
        self.__optimizer = val

    #gradient_weights getter
    def gradient_weights(self):
        return self.__gradient_weights

    #gradient_weight setter
    def gradient_weights(self, val):
        self.__gradient_weights = val

    def forward(self, input_tensor):
        # Bias vector: One bias per data point
        bias_list = [1] * input_tensor.shape[0]
        # To concatenate with the input_tensor, reshape the bias list to (batch_size,1)
        bias_array = np.array(bias_list).reshape(input_tensor.shape[0], 1)
        ## Concatenate bias terms to the input tensor and initialize the class variable, which is used in the backward pass
        self.input_tensor = np.concatenate((input_tensor, bias_array), axis=1)
        # Compute the dot product of the input and the weights and store it as output of the forward pass
        output_tensor = np.dot(self.input_tensor, self.weights)
        # Check if the shape of the o/p matches the i/p
        assert (output_tensor.shape == (self.input_tensor.shape[0], self.output_size))
        return output_tensor

    # Backward pass implementation for backpropagation
    def backward(self, error_tensor):
        # For the usage of backpropagation in the previous layer the gradients of the inputs to the current layer must be passed on.
        # After calculating the weights, remove bias term (last term in the gradient_input), as we add the bias to the i/p as well as the bias weights.
        gradient_inputs = np.dot(error_tensor, self.weights.T)[:, :-1]
        # The gradient of the loss w.r.t the current layer's parameter is used to optimize these paramters
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        # If only the optimizer is set, optimize the parameter using it.
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return gradient_inputs






