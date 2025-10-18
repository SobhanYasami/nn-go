package nn

import "math/rand"

type Layer struct {
	Weights [][]float64
	Biases  []float64
}

func NewLayer(inputSize, outputSize int) *Layer {
	weights := make([][]float64, outputSize)
	for i := range weights {
		weights[i] = make([]float64, inputSize)
		for j := range weights[i] {
			weights[i][j] = rand.Float64()*2 - 1 // random [-1,1]
		}
	}
	biases := make([]float64, outputSize)
	return &Layer{Weights: weights, Biases: biases}
}
