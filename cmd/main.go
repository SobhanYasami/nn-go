package main

import (
	"fmt"

	dataset "github.com/SobhanYasami/nn-go/internal/data"
	"github.com/SobhanYasami/nn-go/internal/nn"
	"github.com/SobhanYasami/nn-go/internal/utils"
	"github.com/SobhanYasami/nn-go/pkg/logger"
)

func main() {
	log := logger.New("main", logger.DEBUG)

	log.Info("Starting mini neural network demo...")

	// Create dataset
	X, y := dataset.CreateData(100, 3)
	fmt.Println("Generated", len(X), "points")

	if err := utils.PlotData(X, y, 3, "spiral.png"); err != nil {
		panic(err)
	}
	fmt.Println("Plot saved as spiral.png")

	// Create Dense layer with 2 input features and 3 neurons
	layer_1, err := nn.NewDenseLayer(2, 3)
	if err != nil {
		fmt.Println("err in create new layer")
	}

	// Perform Forward pass to next layer
	dl1Output, _ := layer_1.Forward(X)

	fmt.Println("===================")
	fmt.Println("output of layer1:", dl1Output)
	fmt.Println("===================")

	//	Pass output of first layer to Activation function
	af := nn.ActivationFn{}

	_ = af.ReLUInPlace(dl1Output)
	fmt.Println("In-place:", dl1Output) // [[0 3.4 0] [2.1 0 0]]

	fmt.Println("output of layer1:", dl1Output)

	//
	// initialize Numgo
	// ng := &mathx.NumGo{}

	// //? Inputs
	// X := [][]float64{
	// 	{1.0, 2.0, 3.0, 2.5},
	// 	{2.0, 5.0, -1.0, 2.0},
	// 	{-1.5, 2.7, 3.3, -0.8},
	// }

	// //? Weights for layers
	// W := [][]float64{
	// 	{0.2, 0.8, -0.5, 1.0},
	// 	{0.5, -0.91, 0.26, -0.5},
	// 	{-0.26, -0.27, 0.17, 0.87},
	// }
	// W2 := [][]float64{
	// 	{0.1, -0.14, 0.5},
	// 	{-0.5, 0.12, -0.33},
	// 	{-0.44, 0.73, -0.13},
	// }
	// //? Biases
	// B := []float64{2.0, 3.0, 0.5}
	// B2 := []float64{-1.0, 2.0, -0.5}

	// // First Layer
	// batchW1, err := ng.DotMatrix(X, ng.Transpose(W))
	// if err != nil {
	// 	log.Error("DotMatrix failed:", err)
	// 	return
	// }
	// // Add biases
	// output1 := make([][]float64, len(batchW1))
	// for i, op := range batchW1 {
	// 	output1[i], _ = ng.AddVectors(op, B)
	// }

	// // Second Layer
	// batchW2, err := ng.DotMatrix(output1, ng.Transpose(W2))
	// if err != nil {
	// 	log.Error("DotMatrix failed:", err)
	// 	return
	// }
	// output2 := make([][]float64, len(batchW2))
	// for i, op := range batchW2 {
	// 	output2[i], _ = ng.AddVectors(op, B2)
	// }

	// fmt.Println("output of first layer is:\n", output1)
	// fmt.Println("===========================================")
	// fmt.Println("output of second layer is:\n", output2)

}
