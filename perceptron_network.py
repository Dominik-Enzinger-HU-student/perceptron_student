

class Perceptron():
    """The perceptron class."""
    def __init__(self, weights):
        """Initializes the perceptron.
        
        Args:
            number_of_weights (int)): The number of weights the perceptron should have.
        """
        self.weights: list[float] = weights  # The bias is represented as the first weight (w[0])
    
    def __repr__(self):
        """Defines the way the perceptron is represented"""
        return f"Perceptron: Weights: {self.weights}"
    
    def output(self, inputs: list[float]):
        """The output function of the perceptron, using the step function as activation function.
        
        Args:
            inputs (list[float]): A list of inputs. x[0] should be 1.
        """
        dot_product = sum(inputs[i] * self.weights[i] for i in range(len(inputs)))
        if dot_product >= 0:
            output = 1
        else:
            output = 0
        return output
    
    def train(self, training_inputs: list[list[float]], targets: list[float], learning_rate: float = 0.1):
        """Trains the perceptrons weights with the given training inputs and expected outputs (targets).
        
        Args:
            training_inputs (list[list[float]]): A two dimensional list containing all the possible inputs.
            targets (list[float]): A list containing the expected outputs for each input.
            learning_rate (float): The learning rate (n, eta) of the perceptron.
        """
        counter = 0  # Initialize counter for training passes
        while self.loss(training_inputs, targets) != 0:
            if counter >= 10000:
                print(f"Training stopped: No solution found after {counter} passes.")
                break
            counter += 1
            for i, _ in enumerate(training_inputs):
                self.update(training_inputs[i], targets[i], learning_rate)
    
    def update(self, training_input: list[float], target: float, learning_rate: float = 0.1):
        """Applies the learning rule to the perceptrons weights.
        
        Args:
            training_input (list[float]): A list containing the training input.
            target (float): The corresponding target for the training input.
            learning_rate (float): The learning rage (n, eta) of the perceptron.
        """
        output = self.output(training_input)
        error = target - output
        for j, _ in enumerate(self.weights):
            delta_weight = learning_rate * error * training_input[j]
            self.weights[j] = self.weights[j] + delta_weight

    def loss(self, training_inputs: list[list[float]], targets: list[float]):
        """Calculates the MSE
        
        Args:
            training_inputs (list[list[float]]): A two dimensional list containing the training inputs.
            targets (list[float]): A list containing the corresponding targets for the inputs.
        """
        number_of_training_inputs = len(training_inputs)
        sum_squared_errors = 0
        for i, _ in enumerate(training_inputs):
            output = self.output(training_inputs[i])
            error = targets[i] - output
            squared_error = error ** 2
            sum_squared_errors += squared_error
        mse = sum_squared_errors / number_of_training_inputs
        return mse
    

class PerceptronLayer():
    def __init__(self, weight_vectors : list[list]):
        """Initializes a perceptron layer.

        Args:
            weight_vectors: A two dimensional list containg the weights for each perceptron.
        """
        self.weight_vectors = weight_vectors
        self.perceptrons = [Perceptron(weights) for weights in weight_vectors]

    def __str__(self):
        """Defines the way the PerceptronLayer is printed."""
        return f"Layer: {self.perceptrons}"
    
    def __repr__(self):
        """Defines the way the PerceptronLayer is represented."""
        return self.__str__()
        
    def outputs(self, inputs):
        """Calculates the output of each perceptron in the layer and returns them as a list."""
        output_array = []  # List containing the outputs of the perceptrons
        for perceptron in self.perceptrons:
            output = perceptron.output(inputs)
            output_array.append(output)
        return output_array
    

class PerceptronNetwork():
    def __init__(self, layer_vectors : list[list[list]]):
        """Initializes a perceptron Network.
        
        Args:
            layer_vectors: A three dimensional list containing the weights for each perceptron for each layer.
        """
        self.layer_vectors = layer_vectors
        self.layers = [PerceptronLayer(weight_vectors) for weight_vectors in layer_vectors]

    def __str__(self):
        """Defines the way the PerceptronNetwork is represented."""
        return f"Network: {self.layers}"
    
    def __repr__(self):
        return self.__str__()
    
    def output(self, inputs):
        """Calculates the output of the Network."""
        for layer in self.layers:
            outputs = layer.outputs(inputs)            
            inputs = [1] + outputs  # The outputs become the new inputs for the next layer. Prepent a 1 at the start of each new input layer for the calculation with the bias.
        return outputs