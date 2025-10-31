# `mathx` Package Documentation

This document describes the **`mathx`** package, which provides foundational **numerical** and **linear algebra** operations for building neural network and numerical computation systems in Go.

---

## 📦 Overview

**Package:** `mathx`
**Purpose:** Implements lightweight, dependency-free mathematical operations similar to **NumPy** functions (e.g., `np.dot`, `np.maximum`, `np.linalg.norm`).

**Main type:**

```go
type NumGo struct{}
```

All instance methods are attached to this struct to mimic NumPy-like organization.

---

## 🧮 Scalar Operations

### **RoundTo**

```go
func (ng *NumGo) RoundTo(x float64, digits int) float64
```

Rounds a float64 value `x` to the specified number of decimal digits.

| Parameter | Type    | Description              |
| --------- | ------- | ------------------------ |
| `x`       | float64 | The value to round       |
| `digits`  | int     | Number of decimal digits |

**Example**

```go
ng := mathx.NumGo{}
fmt.Println(ng.RoundTo(3.14159, 2)) // 3.14
```

---

## 🧭 Vector Operations

### **DotVectors**

```go
func (ng *NumGo) DotVectors(v1, v2 []float64) (float64, error)
```

Computes the **dot product** of two equal-length vectors.

[
v_1 \cdot v_2 = \sum_i v_1[i] \times v_2[i]
]

---

### **AddVectors**

```go
func (ng *NumGo) AddVectors(v1, v2 []float64) ([]float64, error)
```

Performs element-wise addition:
[
\text{result}\_i = v_1[i] + v_2[i]
]

---

### **SubVectors**

```go
func (ng *NumGo) SubVectors(v1, v2 []float64) ([]float64, error)
```

Performs element-wise subtraction:
[
\text{result}\_i = v_1[i] - v_2[i]
]

---

### **ScaleVector**

```go
func (ng *NumGo) ScaleVector(v []float64, scalar float64) []float64
```

Multiplies all elements by a scalar value.

---

### **Norm**

```go
func (ng *NumGo) Norm(v []float64) float64
```

Computes the **Euclidean (L2) norm** of a vector:
[
|v| = \sqrt{\sum_i v_i^2}
]

---

### **Normalize**

```go
func (ng *NumGo) Normalize(v []float64) []float64
```

Returns a **unit vector** by dividing by the norm.
If the vector norm is zero, returns a zero vector of the same length.

---

### **MaxVector**

```go
func MaxVector(a, b []float64) ([]float64, error)
```

Performs elementwise maximum operation, equivalent to **`np.maximum(a, b)`**.

[
\text{result}_i = \max(a_i, b_i)
]

**Example**

```go
a := []float64{1, -2, 3}
b := []float64{0, 5, -1}
res, _ := mathx.MaxVector(a, b)
fmt.Println(res) // [1 5 3]
```

---

## 🧩 Matrix Operations

### **DotMatrix**

```go
func (ng *NumGo) DotMatrix(A, B [][]float64) ([][]float64, error)
```

Performs standard **matrix multiplication** (A*{m×n} \times B*{n×p} = C\_{m×p}).

**Requirements:**

- Number of columns in `A` must match number of rows in `B`.

**Example**

```go
A := [][]float64{{1, 2}, {3, 4}}
B := [][]float64{{5, 6}, {7, 8}}
ng := mathx.NumGo{}
C, _ := ng.DotMatrix(A, B)
// C = [[19, 22], [43, 50]]
```

---

### **Transpose**

```go
func (ng *NumGo) Transpose(M [][]float64) [][]float64
```

Computes the **transpose** of a matrix ( M*{m×n} → M^T*{n×m} ).

**Example**

```go
M := [][]float64{{1, 2, 3}, {4, 5, 6}}
ng := mathx.NumGo{}
T := ng.Transpose(M)
// T = [[1, 4], [2, 5], [3, 6]]
```

---

### **MaxMatrix**

```go
func MaxMatrix(a, b [][]float64) ([][]float64, error)
```

Returns an elementwise maximum between two 2D matrices, similar to **`np.maximum(A, B)`**.

**Example**

```go
A := [][]float64{{1, -2}, {3, 0}}
B := [][]float64{{0, 5}, {-1, 2}}
C, _ := mathx.MaxMatrix(A, B)
// C = [[1, 5], [3, 2]]
```

---

## ⚙️ Error Handling

All vector and matrix operations validate shape compatibility.
If dimensions mismatch, they return a descriptive `error`.

---

## 📘 Usage Example

```go
package main

import (
	"fmt"
	"github.com/SobhanYasami/nn-go/internal/mathx"
)

func main() {
	ng := mathx.NumGo{}

	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}

	dot, _ := ng.DotVectors(a, b)
	fmt.Println("Dot product:", dot)

	norm := ng.Norm(a)
	fmt.Println("Norm:", norm)

	max, _ := mathx.MaxVector(a, b)
	fmt.Println("Max vector:", max)
}
```

**Output**

```
Dot product: 32
Norm: 3.741657
Max vector: [4 5 6]
```

---

## 📚 References

- [NumPy Linear Algebra Documentation](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [Matrix Multiplication — Wikipedia](https://en.wikipedia.org/wiki/Matrix_multiplication)
- [Golang `math` Package Docs](https://pkg.go.dev/math)
