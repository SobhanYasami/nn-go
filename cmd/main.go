package main

import (
	"fmt"

	"github.com/SobhanYasami/nn-go/internal/nn"
)

func main() {
	// Example input batch (3 samples, 4 features)
	inputs := [][]float64{
		{1.0, 2.0, 3.0, 2.5},
		{2.0, 5.0, -1.0, 2.0},
		{-1.5, 2.7, 3.3, -0.8},
	}

	model := nn.NewModel()
	model.AddLayer(4, 3) // input_size=4, output_size=3
	model.AddLayer(3, 3) // second layer

	outputs := model.Forward(inputs)
	fmt.Println("Final outputs:", outputs)
}
