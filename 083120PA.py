# DataCamp

# Introduction to Deep Learning with PyTorch

# Creating tensors in PyTorch

# Import torch
import torch

# Create random tensor of size 3 by 3
your_first_tensor = torch.randn(3, 3)

# Calculate the shape of the tensor
tensor_size = your_first_tensor.size()

# Print the values of the tensor and its shape
print(your_first_tensor)
print(tensor_size)

# Matrix multiplication

# Create a matrix of ones with shape 3 by 3
tensor_of_ones = torch.ones(3, 3)

# Create an identity matrix with shape 3 by 3
identity_tensor = torch.eye(3)

# Do a matrix multiplication of tensor_of_ones with identity_tensor
matrices_multiplied = torch.mm(tensor_of_ones, identity_tensor)
print(matrices_multiplied)

# Do an element-wise multiplication of tensor_of_ones with identity_tensor
element_multiplication = tensor_of_ones * identity_tensor
print(element_multiplication)