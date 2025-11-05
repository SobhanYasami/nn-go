package main

import (
	"fmt"
	"math/rand"
	"time"

	dataset "github.com/SobhanYasami/nn-go/internal/data"
	"github.com/SobhanYasami/nn-go/internal/nn"
	"github.com/SobhanYasami/nn-go/internal/utils"
	"github.com/SobhanYasami/nn-go/pkg/logger"
)

func main() {
	log := logger.New("main", logger.DEBUG)
	log.Info("Starting hill climbing neural network demo...")

	start := time.Now()

	// --- Step 1: Create dataset ---
	X, y := dataset.CreateData(300, 3)
	fmt.Println("Generated", len(X), "points")

	if err := utils.PlotData(X, y, 3, "spiral.png"); err != nil {
		panic(err)
	}
	fmt.Println("✅ Dataset ready.")

	// --- Step 2: Initialize activation and loss functions ---
	af := nn.ActivationFn{}
	lf := nn.LossFn{}

	// --- Step 3: Define network architecture ---
	layerSizes := []int{2, 8, 8, 6, 6, 4, 3}
	layers := make([]*nn.DenseLayer, len(layerSizes)-1)
	for i := 0; i < len(layerSizes)-1; i++ {
		layer, err := nn.NewDenseLayer(layerSizes[i], layerSizes[i+1])
		if err != nil {
			log.Error("Error creating layer %d: %v", i, err)
			return
		}
		layers[i] = layer
	}

	// --- Step 4: Evaluate initial loss ---
	initialLoss := computeLoss(X, y, layers, &af, &lf)
	bestLoss := initialLoss
	fmt.Printf("Initial loss: %.6f\n", bestLoss)

	// --- Step 5: Hill climbing optimization ---
	rand.Seed(time.Now().UnixNano())
	// bestWeights := cloneNetwork(layers)

	numIterations := 1000
	fmt.Printf("🚀 Running hill climbing for %d iterations...\n", numIterations)

	learningRate := 0.01
	epochs := 10000

	for epoch := 0; epoch < epochs; epoch++ {
		// Forward pass
		input := X
		for i, layer := range layers {
			output, _ := layer.Forward(input)
			if i < len(layers)-1 {
				_ = af.ReLUInPlace(output)
			}
			input = output
		}
		_ = af.SoftmaxInPlace(input)

		// Compute loss
		loss, _ := lf.CategoricalCrossEntropy(input, y)

		// Backward pass
		dInputs := lf.SoftmaxCrossEntropyBackward(input, y)
		for i := len(layers) - 1; i >= 0; i-- {
			if i < len(layers)-1 {
				prev := layers[i].Output
				dInputs = af.ReLUBackward(dInputs, prev)
			}
			dInputs = layers[i].Backward(dInputs, learningRate)
		}

		if epoch%100 == 0 {
			fmt.Printf("Epoch %4d | Loss: %.6f\n", epoch, loss)
		}
	}

	fmt.Printf("\n🏁 Optimization completed.\n")
	fmt.Printf("🔹 Best loss: %.6f\n", bestLoss)
	fmt.Printf("⏱️ Total runtime: %v\n", time.Since(start))
}

// ----------------- Helper functions -----------------

func computeLoss(X [][]float64, y []int, layers []*nn.DenseLayer, af *nn.ActivationFn, lf *nn.LossFn) float64 {
	input := X
	for i, layer := range layers {
		output, _ := layer.Forward(input)
		if i < len(layers)-1 {
			_ = af.ReLUInPlace(output)
		}
		input = output
	}
	_ = af.SoftmaxInPlace(input)
	loss, _ := lf.CategoricalCrossEntropy(input, y)
	return loss
}
