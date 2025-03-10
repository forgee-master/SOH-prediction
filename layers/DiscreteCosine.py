import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from scipy.fft import dct, idct
import math


class DiscreteCosineTransform(Function):
    """
    Implements a custom autograd function for Discrete Cosine Transform (DCT).
    """
    @staticmethod
    def forward(ctx, input):
        # Convert PyTorch tensor to NumPy array for scipy operations
        input_np = input.cpu().numpy()
        
        # Apply DCT (Type-II) with orthonormalization
        transformed_np = dct(input_np, type=2, norm="ortho", axis=-1)
        
        # Convert back to PyTorch tensor and return
        output = torch.from_numpy(transformed_np).to(input.device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Convert gradient to NumPy array
        grad_output_np = grad_output.cpu().numpy()
        
        # Apply IDCT (Type-II) with orthonormalization
        grad_input_np = idct(grad_output_np, type=2, norm='ortho', axis=-1)
        
        # Convert back to PyTorch tensor and return
        grad_input = torch.from_numpy(grad_input_np).to(grad_output.device)
        return grad_input
    

class iDiscreteCosineTransform(Function):
    """
    Implements a custom autograd function for Inverse Discrete Cosine Transform (iDCT).
    """
    @staticmethod
    def forward(ctx, input):
        # Convert PyTorch tensor to NumPy array
        input_np = input.cpu().numpy()

        # Apply IDCT using scipy
        transformed_np = idct(input_np, type=2, axis=-1)

        # Convert back to PyTorch tensor
        output = torch.from_numpy(transformed_np).to(input.device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Convert gradient to NumPy array
        grad_output_np = grad_output.cpu().numpy()

        # Apply DCT using scipy
        grad_input_np = dct(grad_output_np, type=2, axis=-1)
        
        # Convert back to PyTorch tensor
        grad_input = torch.from_numpy(grad_input_np).to(grad_output.device)
        return grad_input   