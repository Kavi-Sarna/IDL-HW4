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
        # TODO: Implement forward pass
        
        # Store input for backward pass
        self.A = A

        self.N = np.prod(A.shape[:-1])  # Batch size
        self.C0 = A.shape[-1]  # Number of input features

        # Reshape A to ensure it's 2D
        A_2D = A.reshape(self.N, self.C0)

        # Compute Z using matrix multiplication and broadcasting
        Z = np.dot(A_2D, self.W.T) + self.b  # Broadcasting b to match A's shape
        # Reshape Z to match the expected output shape
        if len(A.shape) > 2:
            Z = Z.reshape(*A.shape[:-1], self.W.shape[0])
        
        return Z
        # raise NotImplementedError

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass
        
        batch_size = np.prod(self.A.shape[:-1])
        in_features = self.A.shape[-1]

        A_2d = self.A.reshape(batch_size, in_features)

        # Reshape dLdZ to 2D: (batch_size, out_features)
        dLdZ_2D = dLdZ.reshape(batch_size, -1)

        # Compute gradients (refer to the equations in the writeup)
        # self.dLdA = NotImplementedError
        self.dLdA = np.dot(dLdZ_2D, self.W)
        # self.dLdW = NotImplementedError
        self.dLdW = np.dot(dLdZ_2D.T, A_2d)
        # self.dLdb = NotImplementedError
        self.dLdb = np.sum(dLdZ_2D, axis=0)
        # self.dLdA = NotImplementedError
        self.dLdA = self.dLdA.reshape(*self.A.shape)
        
        # Return gradient of loss wrt input
        # raise NotImplementedError
        return self.dLdA
