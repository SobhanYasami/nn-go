# Neural Network from Scratch in Go

## 🧠 Introduction

This package is written in **Go (Golang)** and implements a **neural network from scratch** — without using any external machine learning frameworks.  
It’s designed to help you understand the mathematical foundations behind feed-forward neural networks, matrix operations, and forward propagation.

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
Finally, an **activation function** (such as ReLU, sigmoid, or softmax) can be applied to introduce non-linearity.

---

## 🧩 Example Architecture

Input (4 features)
↓
Dense Layer (3 neurons)
↓
Dense Layer (3 neurons)
↓
Output

Each layer performs a matrix multiplication followed by bias addition and optional activation.

---

## 🚀 Quick Start

```bash
# Clone this repository
git clone https://github.com/SobhanYasami/nn-go.git

cd nn-go

# Run example
go run ./cmd


```
