# Loss Functions Module (`nn/loss.go`)

This document provides documentation for the **loss functions** implemented in the `nn` package.
It includes the **Categorical Cross-Entropy** loss and its corresponding **backward pass** used for gradient computation during training.

---

## 📦 Overview

**Package:** `nn`
**Struct:** `LossFn`
**Purpose:** Implements forward and backward operations for loss calculation in classification neural networks.

```go
type LossFn struct{}
```

---

## ⚙️ 1. Categorical Cross-Entropy Loss

### **Function Signature**

```go
func (lf *LossFn) CategoricalCrossEntropy(predictions [][]float64, yTrue []int) (float64, error)
```

### **Description**

Computes the **mean categorical cross-entropy loss** between model predictions and true labels.
It measures how far the predicted probability distribution is from the true distribution.

### **Mathematical Formula**

[
L = -\frac{1}{N} \sum_{i=1}^{N} \log(p_{i, class_true})
]

Where:

- ( p\_{i, class_true} ) = predicted probability of the correct class for sample _i_
- ( N ) = number of samples

---

### **Arguments**

| Name          | Type          | Description                                                       |
| ------------- | ------------- | ----------------------------------------------------------------- |
| `predictions` | `[][]float64` | 2D slice containing softmax probabilities (each row = one sample) |
| `yTrue`       | `[]int`       | True class indices (e.g. `[0, 2, 1]`)                             |

---

### **Returns**

| Name       | Type      | Description                                                |
| ---------- | --------- | ---------------------------------------------------------- |
| `meanLoss` | `float64` | Average cross-entropy loss across all samples              |
| `error`    | `error`   | Returned if dimensions are invalid or indices out of range |

---

### **Implementation Details**

- Adds a small constant (`epsilon = 1e-15`) to prevent `log(0)` numerical errors.
- Verifies shape consistency between `predictions` and `yTrue`.
- Computes the **mean** loss across the batch.

---

### **Example**

```go
predictions := [][]float64{
    {0.7, 0.2, 0.1},
    {0.1, 0.8, 0.1},
}
yTrue := []int{0, 1}

lf := nn.LossFn{}
loss, err := lf.CategoricalCrossEntropy(predictions, yTrue)
fmt.Println("Loss:", loss)
// Output: Loss ≈ 0.164 (depends on data)
```

---

## 🔙 2. Backpropagation — Softmax + Cross-Entropy Derivative

### **Function Signature**

```go
func (lf *LossFn) SoftmaxCrossEntropyBackward(predictions [][]float64, yTrue []int) [][]float64
```

### **Description**

Computes the **gradient of the loss with respect to the inputs** (logits or softmax outputs) for the **softmax + cross-entropy** combination.

This is the derivative used during **backpropagation** to update network weights.

---

### **Mathematical Derivation**

When combining **softmax** and **cross-entropy**, the gradient simplifies to:

[
\frac{\partial L}{\partial z_i} = \frac{1}{N} (p_i - y_i)
]

Where:

- ( p_i ) = predicted probability vector
- ( y_i ) = one-hot encoded true label
- ( N ) = number of samples

---

### **Returns**

- A 2D slice `[][]float64` with the same shape as `predictions`, representing the gradient for each output neuron.

---

### **Implementation Details**

- Creates a copy of the predictions.
- Subtracts 1 from the predicted probability of the true class.
- Divides by the batch size to normalize.

---

### **Example**

```go
predictions := [][]float64{
    {0.7, 0.2, 0.1},
    {0.1, 0.8, 0.1},
}
yTrue := []int{0, 1}

lf := nn.LossFn{}
grads := lf.SoftmaxCrossEntropyBackward(predictions, yTrue)

fmt.Println(grads)
// Output (approx):
// [[-0.15, 0.10, 0.05],
//  [0.05, -0.10, 0.05]]
```

---

## ⚠️ Notes

- The backward pass assumes that the **forward layer output is already passed through softmax**.
- The returned gradients can be directly passed into previous layer’s `Backward()` method.
- These implementations are suitable for small- to medium-scale experiments (not optimized for GPU).

---

## 📚 References

- [Deep Learning Book – Chapter 6 (Goodfellow et al., 2016)](https://www.deeplearningbook.org/)
- [Stanford CS231n: Softmax Classifier](https://cs231n.github.io/linear-classify/#softmax)
- [Cross-Entropy Loss – Wikipedia](https://en.wikipedia.org/wiki/Cross_entropy)
