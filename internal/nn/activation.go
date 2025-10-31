package nn

import (
	"fmt"
	"math"
)

type ActivationFn struct{}

//? ------------------------------
//? ReLU Activations
//? ------------------------------

// ReLUInPlace applies ReLU(x) = max(0, x) elementwise in-place.
func (af *ActivationFn) ReLUInPlace(inputs [][]float64) error {
	if len(inputs) == 0 {
		return fmt.Errorf("input cannot be empty")
	}
	for i := range inputs {
		for j := range inputs[i] {
			if inputs[i][j] < 0 {
				inputs[i][j] = 0
			}
		}
	}
	return nil
}

// ReLU returns a new slice (non-mutating version).
func (af *ActivationFn) ReLU(inputs [][]float64) ([][]float64, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("input cannot be empty")
	}
	output := make([][]float64, len(inputs))
	for i := range inputs {
		output[i] = make([]float64, len(inputs[i]))
		for j := range inputs[i] {
			if inputs[i][j] > 0 {
				output[i][j] = inputs[i][j]
			}
		}
	}
	return output, nil
}

//? ------------------------------
//? Softmax Activations
//? ------------------------------

// SoftmaxInPlace applies the softmax activation function across each row, in-place.
// Each row in `inputs` is treated as one sample (e.g., output of a layer before activation).
//
// It is numerically stabilized using the max subtraction technique:
// softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x))).
func (af *ActivationFn) SoftmaxInPlace(inputs [][]float64) error {
	if len(inputs) == 0 {
		return fmt.Errorf("input cannot be empty")
	}

	for i := range inputs {
		row := inputs[i]
		if len(row) == 0 {
			continue
		}

		// 1. Find max for numerical stability
		maxVal := row[0]
		for _, v := range row[1:] {
			if v > maxVal {
				maxVal = v
			}
		}

		// 2. Compute exp(x - max)
		var sum float64
		for j, v := range row {
			expv := math.Exp(v - maxVal)
			row[j] = expv
			sum += expv
		}

		// 3. Normalize by sum
		for j := range row {
			row[j] /= sum
		}
	}
	return nil
}

// Softmax returns a new slice (non-mutating version) for functional use.
// Each row in `inputs` is treated as one sample.
func (af *ActivationFn) Softmax(inputs [][]float64) ([][]float64, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("input cannot be empty")
	}

	output := make([][]float64, len(inputs))
	for i := range inputs {
		row := inputs[i]
		if len(row) == 0 {
			continue
		}

		// Find max for stability
		maxVal := row[0]
		for _, v := range row[1:] {
			if v > maxVal {
				maxVal = v
			}
		}

		// Compute exp(x - max)
		output[i] = make([]float64, len(row))
		var sum float64
		for j, v := range row {
			expv := math.Exp(v - maxVal)
			output[i][j] = expv
			sum += expv
		}

		// Normalize
		for j := range output[i] {
			output[i][j] /= sum
		}
	}
	return output, nil
}

func (af *ActivationFn) ReLUBackward(dOutputs, inputs [][]float64) [][]float64 {
	dInputs := make([][]float64, len(dOutputs))
	for i := range dOutputs {
		dInputs[i] = make([]float64, len(dOutputs[i]))
		for j := range dOutputs[i] {
			if inputs[i][j] > 0 {
				dInputs[i][j] = dOutputs[i][j]
			}
		}
	}
	return dInputs
}
