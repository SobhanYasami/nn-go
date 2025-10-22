package mathx

import (
	"errors"
	"math"
)

// NumGo provides basic numerical and linear algebra operations.
type NumGo struct{}

//? --------------------
//? Scalar Operations
//? --------------------

// RoundTo rounds a float64 value x to the specified number of decimal digits.
func (ng *NumGo) RoundTo(x float64, digits int) float64 {
	if digits < 0 {
		return math.Round(x)
	}
	pow := math.Pow(10, float64(digits))
	return math.Round(x*pow) / pow
}

//? --------------------
//? Vector Operations
//? --------------------

// DotVectors computes the dot product of two vectors.
func (ng *NumGo) DotVectors(v1, v2 []float64) (float64, error) {
	if len(v1) != len(v2) {
		return 0, errors.New("DotVectors: vector length mismatch")
	}
	var result float64
	for i := range v1 {
		result += v1[i] * v2[i]
	}
	return result, nil
}

// AddVectors returns the element-wise sum of two vectors.
func (ng *NumGo) AddVectors(v1, v2 []float64) ([]float64, error) {
	if len(v1) != len(v2) {
		return nil, errors.New("AddVectors: vector length mismatch")
	}
	result := make([]float64, len(v1))
	for i := range v1 {
		result[i] = v1[i] + v2[i]
	}
	return result, nil
}

// SubVectors returns the element-wise difference of two vectors.
func (ng *NumGo) SubVectors(v1, v2 []float64) ([]float64, error) {
	if len(v1) != len(v2) {
		return nil, errors.New("SubVectors: vector length mismatch")
	}
	result := make([]float64, len(v1))
	for i := range v1 {
		result[i] = v1[i] - v2[i]
	}
	return result, nil
}

// ScaleVector scales all elements of a vector by a constant factor.
func (ng *NumGo) ScaleVector(v []float64, scalar float64) []float64 {
	result := make([]float64, len(v))
	for i := range v {
		result[i] = v[i] * scalar
	}
	return result
}

// Norm returns the Euclidean (L2) norm of a vector.
func (ng *NumGo) Norm(v []float64) float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

// Normalize returns a normalized version of the vector (unit vector).
func (ng *NumGo) Normalize(v []float64) []float64 {
	norm := ng.Norm(v)
	if norm == 0 {
		return make([]float64, len(v))
	}
	result := make([]float64, len(v))
	for i := range v {
		result[i] = v[i] / norm
	}
	return result
}

// MaxVector returns elementwise maximum between two float64 slices (like np.maximum).
func MaxVector(a, b []float64) ([]float64, error) {
	if len(a) != len(b) {
		return nil, errors.New("vectors must have the same length")
	}

	result := make([]float64, len(a))
	for i := range a {
		if a[i] > b[i] {
			result[i] = a[i]
		} else {
			result[i] = b[i]
		}
	}
	return result, nil
}

//? --------------------
//? Matrix Operations
//? --------------------

// MatrixMul performs matrix multiplication A (m×n) * B (n×p) = C (m×p).
func (ng *NumGo) DotMatrix(A, B [][]float64) ([][]float64, error) {
	if len(A) == 0 || len(B) == 0 {
		return nil, errors.New("MatrixMul: empty matrix")
	}
	n, p := len(B), len(B[0])
	if len(A[0]) != n {
		return nil, errors.New("MatrixMul: incompatible dimensions")
	}

	m := len(A)
	C := make([][]float64, m)
	for i := range C {
		C[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			for k := 0; k < n; k++ {
				C[i][j] += A[i][k] * B[k][j]
			}
		}
	}
	return C, nil
}

// Transpose returns the transpose of a matrix.
func (ng *NumGo) Transpose(M [][]float64) [][]float64 {
	if len(M) == 0 {
		return [][]float64{}
	}
	rows, cols := len(M), len(M[0])
	T := make([][]float64, cols)
	for i := range T {
		T[i] = make([]float64, rows)
		for j := range M {
			T[i][j] = M[j][i]
		}
	}
	return T
}

// MaxMatrix returns elementwise maximum between two 2D slices (matrices).
func MaxMatrix(a, b [][]float64) ([][]float64, error) {
	if len(a) != len(b) {
		return nil, errors.New("matrices must have the same number of rows")
	}

	result := make([][]float64, len(a))
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return nil, errors.New("matrices must have the same number of columns")
		}
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			if a[i][j] > b[i][j] {
				result[i][j] = a[i][j]
			} else {
				result[i][j] = b[i][j]
			}
		}
	}
	return result, nil
}
