package nn

import (
	"errors"
	"math/rand"
	"time"
)

// ? DenseLayer represents a fully connected layer.
type DenseLayer struct {
	Weights [][]float64
	Biases  []float64

	//! Cache for backpropagation
	Input    [][]float64
	Output   [][]float64
	DWeights [][]float64
	DBiases  []float64
}

// ? NewDenseLayer creates a new dense layer.
func NewDenseLayer(nInputs, nNeurons int) (*DenseLayer, error) {
	if nInputs <= 0 || nNeurons <= 0 {
		return nil, errors.New("inputs and neurons must be positive")
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	weights := make([][]float64, nNeurons)
	for i := range weights {
		weights[i] = make([]float64, nInputs)
		for j := range weights[i] {
			weights[i][j] = rng.NormFloat64() * 0.01
		}
	}

	biases := make([]float64, nNeurons)

	return &DenseLayer{
		Weights: weights,
		Biases:  biases,
	}, nil
}

// ?
//
//	Forward pass: store inputs and outputs for backprop.
//
// ##
func (dl *DenseLayer) Forward(X [][]float64) ([][]float64, error) {
	if len(X) == 0 {
		return nil, errors.New("empty input")
	}

	dl.Input = X
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
	dl.Output = output
	return output, nil
}

// ?
// Backward pass: compute gradients.
// ##
func (dl *DenseLayer) Backward(dOutputs [][]float64, learningRate float64) [][]float64 {
	batchSize := float64(len(dl.Input))

	// Initialize gradients
	dl.DWeights = make([][]float64, len(dl.Weights))
	dl.DBiases = make([]float64, len(dl.Biases))
	for i := range dl.DWeights {
		dl.DWeights[i] = make([]float64, len(dl.Weights[i]))
	}

	// Compute dWeights, dBiases
	for i := 0; i < len(dl.Weights); i++ { // each neuron
		for j := 0; j < len(dl.Weights[i]); j++ {
			var grad float64
			for k := 0; k < len(dl.Input); k++ { // batch
				grad += dl.Input[k][j] * dOutputs[k][i]
			}
			dl.DWeights[i][j] = grad / batchSize
		}
	}

	for i := 0; i < len(dl.Biases); i++ {
		var grad float64
		for k := 0; k < len(dOutputs); k++ {
			grad += dOutputs[k][i]
		}
		dl.DBiases[i] = grad / batchSize
	}

	// Compute gradient for inputs
	dInputs := make([][]float64, len(dl.Input))
	for i := range dl.Input {
		dInputs[i] = make([]float64, len(dl.Weights[0]))
		for j := 0; j < len(dl.Weights[0]); j++ {
			for n := 0; n < len(dl.Weights); n++ {
				dInputs[i][j] += dOutputs[i][n] * dl.Weights[n][j]
			}
		}
	}

	// Update weights and biases
	for i := range dl.Weights {
		for j := range dl.Weights[i] {
			dl.Weights[i][j] -= learningRate * dl.DWeights[i][j]
		}
		dl.Biases[i] -= learningRate * dl.DBiases[i]
	}

	return dInputs
}
