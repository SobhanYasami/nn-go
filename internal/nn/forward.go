package nn

import "errors"

// Forward computes the output of the dense layer given an input matrix X.
// Each row of X is an input sample.
func (dl *DenseLayer) Forward(X [][]float64) ([][]float64, error) {
	if len(dl.Weights) == 0 || len(dl.Biases) == 0 {
		return nil, errors.New("layer not initialized")
	}
	if len(X) == 0 || len(X[0]) != len(dl.Weights[0]) {
		return nil, errors.New("input size mismatch")
	}

	output := make([][]float64, len(X))
	for i, sample := range X {
		row := make([]float64, len(dl.Weights))
		for n, weights := range dl.Weights {
			sum := dl.Biases[n]
			for j, w := range weights {
				sum += sample[j] * w
			}
			row[n] = sum
		}
		output[i] = row
	}
	return output, nil
}
