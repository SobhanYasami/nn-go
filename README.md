## Inroduction

this a package written in golang to implement neural network from scratch

## How multi-layer neural network works

X is batach of inputs with (n\*m) dimension which:
n: number of samples
m: number of features

B: bias vector with dimension of n\*1

W_layer1 is matrix for 4 neurons with n\*m so output of first layer is

output1 = dotProduct(X,W_layer1_T)+B1

for second layer we have:
output2 = dotProduct(output1,W_layer2_T) + B2

and so on.
