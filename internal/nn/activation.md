# Activation Functions Module (`nn/activation.go`)

This document describes the **Activation Functions** implemented in the `nn` package using Go.
It provides **ReLU** (Rectified Linear Unit) and **Softmax** activations with both **in-place** and **functional** (non-mutating) variants, as well as a **backward** pass for ReLU.

---

## 📦 Overview

**Package:** `nn`
**Struct:** `ActivationFn`
**Purpose:** To handle activation transformations during forward and backward passes of neural network layers.

```go
type ActivationFn struct{}
```

This struct provides methods to perform:

- ReLU activation (and its derivative)
- Softmax activation with numerical stability

---

## ⚙️ ReLU Activation

### 1. `ReLUInPlace(inputs [][]float64) error`

Applies the **ReLU** activation directly on the input slice.

**Formula:**
[
ReLU(x) = \max(0, x)
]

**Parameters:**

- `inputs`: 2D slice of values (modified in-place).

**Behavior:**

- Negative values are replaced with 0.
- Positive values remain unchanged.

**Returns:**

- `error` if input is empty.

**Example:**

```go
inputs := [][]float64{{-1, 2, -3}, {4, -5, 6}}
af := nn.ActivationFn{}
_ = af.ReLUInPlace(inputs)
// inputs = {{0, 2, 0}, {4, 0, 6}}
```

---

### 2. `ReLU(inputs [][]float64) ([][]float64, error)`

Returns a **new slice** with ReLU applied (non-mutating version).

**Example:**

```go
inputs := [][]float64{{-1, 2}, {3, -4}}
af := nn.ActivationFn{}
out, _ := af.ReLU(inputs)
// out = {{0, 2}, {3, 0}}
```

---

## 🔙 ReLU Backward (Derivative)

### `ReLUBackward(dOutputs, inputs [][]float64) [][]float64`

Computes the derivative of the ReLU activation during **backpropagation**.

**Logic:**

- Gradient is passed through only for positive input values.
- For ( x \le 0 ), gradient becomes 0.

**Parameters:**

- `dOutputs`: Gradient of loss w.r.t. ReLU outputs.
- `inputs`: Original inputs before ReLU activation.

**Returns:**

- Gradient of loss w.r.t. inputs (`dInputs`).

**Example:**

```go
inputs := [][]float64{{-1, 2}, {3, -4}}
dOut := [][]float64{{1, 1}, {1, 1}}
grad := af.ReLUBackward(dOut, inputs)
// grad = {{0, 1}, {1, 0}}
```

---

## 🔥 Softmax Activation

Softmax converts raw scores into **probabilities** that sum to 1 across each row.

**Formula:**
[
softmax(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
]

The subtraction of `max(x)` ensures **numerical stability** by preventing overflow during exponentiation.

---

### 1. `SoftmaxInPlace(inputs [][]float64) error`

Applies softmax **in-place** across each row (sample).

**Behavior:**

- Each row in `inputs` is treated as a single sample.
- The function normalizes values so that the sum of each row equals 1.

**Example:**

```go
inputs := [][]float64{{2.0, 1.0, 0.1}}
af := nn.ActivationFn{}
_ = af.SoftmaxInPlace(inputs)
// inputs = {{0.659, 0.242, 0.098}} (approximately)
```

---

### 2. `Softmax(inputs [][]float64) ([][]float64, error)`

Computes softmax and returns a **new slice** (non-mutating).

**Example:**

```go
inputs := [][]float64{{2.0, 1.0, 0.1}}
af := nn.ActivationFn{}
output, _ := af.Softmax(inputs)
// output = {{0.659, 0.242, 0.098}}
```

---

## ⚠️ Notes

- All methods assume input format:
  `[][]float64` → Each inner slice represents a **sample**.
- Functions return an error if the input batch is empty.
- Uses `math.Exp` and `math` standard library for numerical computation.
- Softmax implementation is **row-wise**, not column-wise.

---

## 🧩 Example Integration

```go
package main

import (
    "fmt"
    "github.com/yourusername/yourrepo/nn"
)

func main() {
    af := nn.ActivationFn{}

    X := [][]float64{{1.0, 2.0, 3.0}}
    reluOut, _ := af.ReLU(X)
    fmt.Println("ReLU Output:", reluOut)

    softmaxOut, _ := af.Softmax(X)
    fmt.Println("Softmax Output:", softmaxOut)
}
```

---

## 📚 References

- [CS231n: Softmax Classifier Notes](https://cs231n.github.io/linear-classify/#softmax)
- [Deep Learning Book – Goodfellow et al. (2016)](https://www.deeplearningbook.org/)
- [Go Standard Library: `math`](https://pkg.go.dev/math)
