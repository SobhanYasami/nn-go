# DenseLayer Package Documentation

This package implements a **fully connected neural network layer** (also known as a _dense layer_) in Go. It supports forward and backward propagation, weight initialization, and gradient-based updates.

---

## 📦 Package Overview

**Package name:** `nn`
**Purpose:** Provides a basic implementation of a dense (fully connected) layer for neural networks.

---

## 🧠 DenseLayer Structure

```go
type DenseLayer struct {
    Weights  [][]float64
    Biases   []float64

    // Cached values for backpropagation
    Input    [][]float64
    Output   [][]float64
    DWeights [][]float64
    DBiases  []float64
}
```

### Fields:

- **Weights:** 2D matrix of connection weights between neurons.
- **Biases:** Bias terms for each neuron.
- **Input:** Input batch stored during forward pass.
- **Output:** Output batch after forward pass.
- **DWeights:** Gradients of weights computed during backpropagation.
- **DBiases:** Gradients of biases computed during backpropagation.

---

## ⚙️ Constructor

### `NewDenseLayer(nInputs, nNeurons int) (*DenseLayer, error)`

Creates and initializes a new dense layer.

- **Parameters:**

  - `nInputs`: Number of input features.
  - `nNeurons`: Number of neurons (output features).

- **Initialization Details:**

  - Weights are initialized using a **normal distribution scaled by 0.01**.
  - Biases are initialized to zero.
  - A random seed is created based on the current system time.

- **Returns:**

  - Pointer to `DenseLayer` instance.
  - Error if inputs or neurons are non-positive.

---

## 🔁 Forward Pass

### `func (dl *DenseLayer) Forward(X [][]float64) ([][]float64, error)`

Performs the forward propagation step.

- **Parameters:**

  - `X`: Batch input as a 2D slice (`batch_size x nInputs`).

- **Process:**

  - Stores the input in `dl.Input`.
  - Computes output as:
    [
    y = X \cdot W^T + b
    ]
  - Stores and returns `dl.Output`.

- **Returns:**

  - Output matrix (`batch_size x nNeurons`).
  - Error if input is empty.

---

## 🔙 Backward Pass

### `func (dl *DenseLayer) Backward(dOutputs [][]float64, learningRate float64) [][]float64`

Computes gradients and updates parameters using backpropagation.

- **Parameters:**

  - `dOutputs`: Gradient of loss with respect to layer output.
  - `learningRate`: Step size for gradient descent.

- **Steps:**

  1. Compute gradients for weights (`DWeights`) and biases (`DBiases`).
  2. Compute gradients for inputs (`dInputs`):
     [
     dX = dY \cdot W
     ]
  3. Update weights and biases using:
     [
     W := W - \eta \cdot DWeights
     ]
     [
     b := b - \eta \cdot DBiases
     ]

- **Returns:**

  - Gradient of loss with respect to the input (`dInputs`).

---

## 🧩 Example Usage

```go
package main

import (
    "fmt"
    "github.com/yourusername/yourrepo/nn"
)

func main() {
    layer, _ := nn.NewDenseLayer(3, 2)

    inputs := [][]float64{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
    }

    outputs, _ := layer.Forward(inputs)
    fmt.Println("Forward Output:", outputs)

    dOutputs := [][]float64{
        {0.1, -0.2},
        {0.05, 0.1},
    }

    dInputs := layer.Backward(dOutputs, 0.01)
    fmt.Println("Backward Input Gradient:", dInputs)
}
```

---

## 🧾 Notes

- This is a simplified educational implementation meant for understanding neural network mechanics in Go.
- Matrix operations are implemented manually without external dependencies.
- Not optimized for GPU or large-scale datasets.

---

## 📚 References

- [Deep Learning Book – Ian Goodfellow et al. (2016)](https://www.deeplearningbook.org/)
- [Go Official Documentation](https://pkg.go.dev/)
- [CS231n: Neural Networks Notes](https://cs231n.github.io/neural-networks-case-study/)
