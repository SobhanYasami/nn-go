package nn

import (
	"fmt"
)

type ActivationFn struct{}

// ReLU applies the Rectified Linear Unit activation function elementwise:
// ReLU(x) = max(0, x)
// It modifies the input in-place for performance.
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

// ReLU returns a new slice (non-mutating version), for safer functional use.
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
