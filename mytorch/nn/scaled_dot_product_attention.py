import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # TODO: Implement forward pass
    
        # Store inputs for backward pass
        self.Q = Q
        self.K = K
        self.V = V
    
        # Calculate attention scores: (N, ..., H, L, S)
        # (N, ..., H, L, E) @ (N, ..., H, E, S) -> (N, ..., H, L, S)
        E = K.shape[-1]
        scaled_dot_product = Q @ K.swapaxes(-2, -1) / np.sqrt(E)
    
        # Apply mask before softmax if provided
        # If mask is not None, add -self.eps to the attention scores for positions to ignore
        if mask is not None:
            scaled_dot_product = np.where(mask, -self.eps, scaled_dot_product)

        # Compute attention scores: Apply softmax along S dimension (N, ..., H, L, S)
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # Calculate output: (N, ..., H, L, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        output = self.attention_scores @ V

        # Return output
        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # Calculate gradients for V: (N, ..., H, S, Ev)
        # (N, ..., H, L, S).transpose(-2, -1) @ (N, ..., H, L, Ev) -> (N, ..., H, S, Ev) 
        d_V = self.attention_scores.swapaxes(-2, -1) @ d_output
    
        # Calculate gradients for attention scores
        # (N, ..., H, L, Ev) @ (N, ..., H, Ev, S) -> (N, ..., H, L, S)
        d_attention_scores = d_output @ self.V.swapaxes(-2, -1)
    
        # Get gradients through softmax
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
    
        # Get embedding dimension for scaling
        E = self.K.shape[-1]
    
        # Calculate gradients for Q: (N, ..., H, L, E)   
        # (N, ..., H, L, S) @ (N, ..., H, S, E) -> (N, ..., H, L, E)
        d_Q = (d_scaled_dot_product / np.sqrt(E)) @ self.K
    
        # Calculate gradients for K: (N, ..., H, S, E)
        # (N, ..., H, S, L) @ (N, ..., H, L, E) -> (N, ..., H, S, E)
        d_K = (d_scaled_dot_product.swapaxes(-2, -1) / np.sqrt(E)) @ self.Q
    
        # Return gradients for Q, K, V
        return d_Q, d_K, d_V