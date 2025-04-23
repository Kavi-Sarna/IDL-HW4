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
        # self.A = NotImplementedError
        Z       = np.array(Z)

        exp_Z   = np.exp(Z - np.max(Z, axis=self.dim, keepdims=True))

        self.A  = exp_Z / np.sum(exp_Z, axis=self.dim, keepdims=True)

        # raise NotImplementedError
        return self.A 

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
           
        # Reshape input to 2D
        if len(shape) > 2:
            # self.A = NotImplementedError
            # dLdA = NotImplementedError
            
            # Get the new shape after moving axis
            new_shape   = np.moveaxis(self.A, self.dim, -1).shape
        
            # Flatten all dimensions except the last one
            A_2         = np.moveaxis(self.A, self.dim, -1).reshape(-1, C)
            dLdA_2      = np.moveaxis(dLdA, self.dim, -1).reshape(-1, C)
        else:
            # If already 2D, just use as is
            A_2         = self.A
            dLdA_2      = dLdA
        
        # Initialize the output gradient
        dLdZ_2          = np.zeros_like(A_2)
    
        # Compute Jacobian-vector product for each sample
        for i in range(A_2.shape[0]):
            # For each slice, the Jacobian J has elements:
            # J_mn = a_m(1-a_m) if m=n, -a_m*a_n if m≠n
            # Direct computation of dL @ J without explicitly forming J
            dLdZ_2[i]  = A_2[i] * (dLdA_2[i] - np.sum(A_2[i] * dLdA_2[i]))

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            # self.A = NotImplementedError
            # dLdZ = NotImplementedError
            dLdZ = np.moveaxis(dLdZ_2.reshape(new_shape), -1, self.dim)
        else:
            dLdZ = dLdZ_2

        # raise NotImplementedError
        return dLdZ
