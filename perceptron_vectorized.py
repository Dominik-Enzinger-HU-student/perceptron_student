import numpy as np

class PerceptronNetworkVectorized():
    def __init__(self, weight_matrices, activation_function=lambda x: 1 if x >= 0 else 0):
        """Initializes a perceptron Network.
        
        Args:
            weight_matrices: A list of numpy arrays.
            activation_function: Function to use as activation function. Else step function is used.
        """
        self.weight_matrices = weight_matrices
        self.activation_function = np.vectorize(activation_function)  # Is initialized as step_function

    def __str__(self):
        """Defines the way the PerceptronNetwork is represented."""
        return f"Matrices: {self.weight_matrices}, Activation function: {self.activation_function}"
    
    def __repr__(self):
        return self.__str__()
    
    def output(self, inputs):
        """Calculates the output of the Network."""
        for matrix in self.weight_matrices:
            results = matrix @ inputs
            output = self.activation_function(results)  # Applies the activation function to the results of the matrix multiplication.
            inputs = output
        return output
    
    def step_function(self, input):
        return 1 if input >= 0 else 0  # Every code base needs one one-liner, sorry.
    
    def test(self):
        matrix = np.array([[1, 2, 3], [-0.5, -0.1, -0.7]])
        print(self.activation_function(matrix))
