import numpy as np

class PerceptronNetworkVectorized():
    def __init__(self, weight_matrices):
        """Initializes a perceptron Network.
        
        Args:
            weight_matrices: A numpy list of matrices.
        """

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