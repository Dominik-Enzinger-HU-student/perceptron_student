import numpy as np

class PerceptronNetworkVectorized():
    def __init__(self, weight_matrices, activation_function=lambda x: 1 if x >= 0 else 0):
        """Initializes a perceptron Network.
        
        Args:
            weight_matrices: A list of numpy arrays.
            activation_function: Function to use as activation function. Else step function is used.
        """
        self.weight_matrices = weight_matrices
        self.activation_function = np.vectorize(activation_function)  # Gets initialized as step_function

    def __str__(self):
        """Defines the way the PerceptronNetwork is represented."""
        return f"Matrices: {self.weight_matrices}, Activation function: {self.activation_function}"
    
    def __repr__(self):
        return self.__str__()
    
    def output(self, inputs):
        """Calculates the output of the Network."""
        for matrix in self.weight_matrices:
            results = matrix @ inputs
            output = self.activation_function(results)  # Applies the activation function to the results of the matrix multiplication. https://www.geeksforgeeks.org/numpy/vectorized-operations-in-numpy/
            # Check if the shape is a vector or a matrix and add 1's for the calculation with the bias.
            if len(np.shape(output)) == 0:  # Its a single number -> done.
                return output
            elif len(np.shape(output)) == 1:  # shape returns a touple (rows, columns). if len == 1 then its a vector.
                output = np.append(1, output)
            else:  # its a matrix.
                n_columns = np.shape(output)[1]  # The number of columns in the matrix.
                vector_to_add = np.ones(n_columns)
                output = np.vstack((vector_to_add, output))  # Thank you Brian.
            inputs = output
        # The output isn't a single number -> top row has to be removed.
        output = np.delete(output, 0, 0)  # Delete the top of the matrix/vector
        return output
