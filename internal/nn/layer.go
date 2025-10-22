package nn

import (
	"errors"
	"math/rand"
	"time"
)

// DenseLayer represents a fully connected layer in a neural network.
type DenseLayer struct {
	Weights [][]float64 // Shape: [n_neurons][n_inputs]
	Biases  []float64   // Shape: [n_neurons]
}

// NewDenseLayer creates a new dense (fully connected) layer.
// nInputs: number of input features per neuron
// nNeurons: number of neurons in this layer
func NewDenseLayer(nInputs, nNeurons int) (*DenseLayer, error) {
	if nInputs <= 0 || nNeurons <= 0 {
		return nil, errors.New("inputs and neurons must be positive")
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	weights := make([][]float64, nNeurons)
	for i := range weights {
		weights[i] = make([]float64, nInputs)
		for j := range weights[i] {
			// Small random values (normal distribution)
			weights[i][j] = rng.NormFloat64() * 0.01
		}
	}

	biases := make([]float64, nNeurons)

	return &DenseLayer{
		Weights: weights,
		Biases:  biases,
	}, nil
}
