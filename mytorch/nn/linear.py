import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        
        # Store input for backward pass
        self.A = A
        input_shape = A.shape
        self.N = np.prod(input_shape[:-1])  # Batch size
        self.C0 = input_shape[-1]  # Number of input features
        A_2D = A.reshape(self.N, self.C0)  # Reshape A to ensure it's 2D
        Z = np.dot(A_2D, self.W.T) + self.b  # Broadcasting b to match A's shape
        # Reshape Z to match the expected output shape
        if len(input_shape) > 2:
            Z = Z.reshape(*input_shape[:-1], self.W.shape[0])      
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        input_shape = self.A.shape
    
        # Reshape inputs and gradients to 2D for computation
        batch_size = np.prod(input_shape[:-1]) 
        in_features = input_shape[-1]
        A_2D = self.A.reshape(batch_size, in_features)
    
        # Reshape dLdZ to 2D: (batch_size, out_features)
        dLdZ_2D = dLdZ.reshape(batch_size, -1)
    
        # Compute gradients
        dLdA_2D = np.dot(dLdZ_2D, self.W)  # (batch_size, in_features)
        self.dLdW = np.dot(dLdZ_2D.T, A_2D)  # (out_features, in_features)
        self.dLdb = np.sum(dLdZ_2D, axis=0)  # (out_features,)
    
        # Reshape dLdA back to match input shape
        dLdA = dLdA_2D.reshape(*input_shape)
    
        return dLdA