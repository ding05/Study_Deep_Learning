# Forward propagation

# Initialize tensors x, y and z
x = torch.rand(1000, 1000)
y = torch.rand(1000, 1000)
z = torch.rand(1000, 1000)

# Multiply x with y
q = torch.mm(x, y)

# Multiply elementwise z with q
f = q * z

mean_f = torch.mean(f)

# Backpropagation by auto-differentiation

# Initialize x, y and z to values 4, -3 and 5
x = torch.tensor(4., requires_grad=True)
y = torch.tensor(-3., requires_grad=True)
z = torch.tensor(5., requires_grad=True)

# Set q to sum of x and y, set f to product of q with z
q = x + y
f = q * z

# Compute the derivatives
f.backward()

# Print the gradients
print("Gradient of x is: " + str(x.grad))
print("Gradient of y is: " + str(y.grad))
print("Gradient of z is: " + str(z.grad))

# Multiply tensors x and y
q = x * y

# Elementwise multiply tensors z with q
f = torch.mm(z, q)

mean_f = torch.mean(f)

# Calculate the gradients
mean_f.backward()

# Introduction to Neural Networks

# Initialize the weights of the neural network
weight_1 = torch.rand(784, 1)
weight_2 = torch.rand(1, 200)

# Multiply input_layer with weight_1
hidden_1 = torch.matmul(input_layer, weight_1)

# Multiply hidden_1 with weight_2
output_layer = torch.matmul(hidden_1, weight_2)
print(output_layer)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Instantiate all 2 linear layers  
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):

        # Use the instantiated layers and return x
        x = self.fc1(x)
        x = self.fc2(x)
        return x