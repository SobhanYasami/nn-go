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
	log.Info("Starting random search neural network demo...")

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

	// --- Step 4: Random search ---
	rand.Seed(time.Now().UnixNano())
	bestLoss := 1e9
	bestIter := -1

	numIterations := 100000
	fmt.Printf("🚀 Running random search for %d iterations...\n", numIterations)

	for iter := 0; iter < numIterations; iter++ {
		// Randomize all weights and biases
		for _, layer := range layers {
			for i := range layer.Weights {
				for j := range layer.Weights[i] {
					layer.Weights[i][j] = rand.NormFloat64() * 0.01 // small Gaussian noise
				}
			}
			for i := range layer.Biases {
				layer.Biases[i] = rand.NormFloat64() * 0.01
			}
		}

		// Forward pass
		input := X
		for i, layer := range layers {
			output, _ := layer.Forward(input)
			if i < len(layers)-1 {
				_ = af.ReLUInPlace(output)
			}
			input = output
		}

		// Apply final softmax
		_ = af.SoftmaxInPlace(input)

		// Compute loss
		loss, err := lf.CategoricalCrossEntropy(input, y)
		if err != nil {
			log.Error("Error computing loss: %v", err)
			return
		}

		// Track best loss
		if loss < bestLoss {
			bestLoss = loss
			bestIter = iter
		}

		// Occasionally print progress
		if iter%5000 == 0 {
			fmt.Printf("Iter %6d | Loss: %.6f | Best so far: %.6f (iter %d)\n",
				iter, loss, bestLoss, bestIter)
		}
	}

	fmt.Printf("\n🏁 Random search completed.\n")
	fmt.Printf("🔹 Best loss: %.6f (found at iteration %d)\n", bestLoss, bestIter)
	fmt.Printf("⏱️ Total runtime: %v\n", time.Since(start))
}
