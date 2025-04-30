import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        #self.A = NotImplementedError
        Z_exp = np.exp(Z - np.max(Z, axis=self.dim, keepdims=True))
        self.A = Z_exp / np.sum(Z_exp, axis=self.dim, keepdims=True)
        return self.A
        
        
    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
    
        # Reshape input to 2D if necessary
        if len(shape) > 2:
            # Move dim to the last position
            A_moved = np.moveaxis(self.A, self.dim, -1)
            dLdA_moved = np.moveaxis(dLdA, self.dim, -1)
        
            # Get the new shape after moving axis
            new_shape = A_moved.shape
        
            # Flatten all dimensions except the last one
            A_2d = A_moved.reshape(-1, C)
            dLdA_2d = dLdA_moved.reshape(-1, C)
        else:
            # If already 2D, just use as is
            A_2d = self.A
            dLdA_2d = dLdA
    
        # Initialize the output gradient
        dLdZ_2d = np.zeros_like(A_2d)
    
        # Compute Jacobian-vector product for each sample
        for i in range(A_2d.shape[0]):
            a = A_2d[i]
            dL = dLdA_2d[i]
        
            # Using the simplified formula from the writeup:
            # For each slice, the Jacobian J has elements:
            # J_mn = a_m(1-a_m) if m=n, -a_m*a_n if m≠n
        
            # Direct computation of dL @ J without explicitly forming J
            dLdZ_2d[i] = a * (dL - np.sum(a * dL))
    
        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Reshape back to the moved-axis shape
            dLdZ_moved = dLdZ_2d.reshape(new_shape)
            # Move the last dimension back to its original position
            dLdZ = np.moveaxis(dLdZ_moved, -1, self.dim)
        else:
            dLdZ = dLdZ_2d
    
        return dLdZ