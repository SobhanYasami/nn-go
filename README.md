# Neural Network from Scratch in Go

## 🧠 Introduction

This package is written in **Go (Golang)** and implements a **neural network from scratch** — without using any external machine learning frameworks.  
It’s designed to help you understand the mathematical foundations behind:

- **Feed-forward neural networks**
- **Matrix operations**
- **Forward and backward propagation**
- **Gradient descent optimization**

---

## ⚙️ How a Multi-Layer Neural Network Works

A neural network consists of **layers of neurons**, each performing a mathematical transformation on the input data.

### 🔢 Inputs

Let **X** be a batch of input samples with shape **(n × m)**:

- **n** → number of samples
- **m** → number of features per sample

### ⚖️ Biases and Weights

Each layer has:

- A **weight matrix** `W` of shape **(k × m)**, where _k_ is the number of neurons in the layer.
- A **bias vector** `B` of shape **(k × 1)**, which shifts the activation values.

### 🔁 Forward Pass

For the **first layer**:
output₁ = dot(X, W₁ᵀ) + B₁

For the **second layer**:
output₂ = dot(output₁, W₂ᵀ) + B₂

This process continues for each layer, passing the output of one layer as the input to the next.  
Finally, an **activation function** (such as ReLU or softmax) is applied to introduce non-linearity.

### 🔄 Backward Pass

During **backpropagation**:

- Gradients of the loss with respect to weights and biases are computed.
- Weights are updated using **gradient descent**:

W := W - learning_rate _ dW
B := B - learning_rate _ dB

This allows the network to learn from errors.

---

## 🧩 Example Architecture

Input (4 features)
↓
Dense Layer (3 neurons, ReLU)
↓
Dense Layer (3 neurons, ReLU)
↓
Output (Softmax)

Each layer performs a **matrix multiplication**, adds biases, and applies an **activation function**.

---

## 📚 References

- [NumPy Linear Algebra Documentation](https://numpy.org/doc/stable/reference/routines.linalg.html) — for understanding matrix operations
- [Matrix Multiplication — Wikipedia](https://en.wikipedia.org/wiki/Matrix_multiplication) — mathematical foundation of forward pass
- [Golang `math` Package Docs](https://pkg.go.dev/math) — Go’s standard math utilities
- [Softmax Function — Wikipedia](https://en.wikipedia.org/wiki/Softmax_function) — for multi-class classification output
- [Cross-Entropy Loss — Wikipedia](https://en.wikipedia.org/wiki/Cross_entropy) — loss function used for classification

---

## 🚀 Quick Start

```bash
# Clone this repository
git clone https://github.com/SobhanYasami/nn-go.git

cd nn-go

# Run the main example
go run ./cmd
```
