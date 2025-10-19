package main

import (
	"fmt"

	"github.com/SobhanYasami/nn-go/internal/nn"
	"github.com/SobhanYasami/nn-go/pkg/logger"
)

func main() {
	log := logger.New("main", logger.DEBUG)

	log.Info("Starting mini neural network demo...")

	// initialize Numgo
	ng := &nn.NumGo{}
	// debug: example  ==========
	v1 := []float64{1, 2, 3}
	v2 := []float64{4, 5, 6}

	dot, _ := ng.DotVectors(v1, v2)
	fmt.Println("Dot:", dot) // 32

	sum, _ := ng.AddVectors(v1, v2)
	fmt.Println("Sum:", sum) // [5 7 9]

	norm := ng.Norm(v1)
	fmt.Println("Norm:", norm) // 3.741...

	A := [][]float64{{1, 2}, {3, 4}}
	B := [][]float64{{5, 6}, {7, 8}}
	C, _ := ng.DotMatrix(A, B)
	fmt.Println("MatrixDotProduct:", C)
	// debug: End of Testing

	// Example input batch (3 samples, 4 features)
	X := [][]float64{
		{1.0, 2.0, 3.0, 2.5},
		{2.0, 5.0, -1.0, 2.0},
		{-1.5, 2.7, 3.3, -0.8},
	}

	model := nn.NewModel()
	model.AddLayer(4, 3) // input_size=4, output_size=3
	model.AddLayer(3, 3) // second layer

	outputs := model.Forward(X)
	log.Debug("Forward pass completed, outputs: %+v", outputs)

	fmt.Println("Final outputs:", outputs)
}
