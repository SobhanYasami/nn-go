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
	bestWeights := cloneNetwork(layers)

	numIterations := 100000
	learningRate := 0.05

	fmt.Printf("🚀 Running hill climbing for %d iterations...\n", numIterations)

	for iter := 0; iter < numIterations; iter++ {
		// Perturb weights and biases slightly
		perturbNetwork(layers, learningRate)

		// Compute new loss
		newLoss := computeLoss(X, y, layers, &af, &lf)

		// Accept or revert
		if newLoss < bestLoss {
			bestLoss = newLoss
			bestWeights = cloneNetwork(layers)
		} else {
			// Revert to previous best
			restoreNetwork(layers, bestWeights)
		}

		// Occasionally print progress
		if iter%5000 == 0 {
			fmt.Printf("Iter %6d | Loss: %.6f | Best so far: %.6f\n",
				iter, newLoss, bestLoss)
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

func perturbNetwork(layers []*nn.DenseLayer, scale float64) {
	for _, layer := range layers {
		for i := range layer.Weights {
			for j := range layer.Weights[i] {
				layer.Weights[i][j] += rand.NormFloat64() * scale
			}
		}
		for i := range layer.Biases {
			layer.Biases[i] += rand.NormFloat64() * scale
		}
	}
}

func cloneNetwork(layers []*nn.DenseLayer) []*nn.DenseLayer {
	clone := make([]*nn.DenseLayer, len(layers))
	for i, layer := range layers {
		newLayer := &nn.DenseLayer{
			Weights: make([][]float64, len(layer.Weights)),
			Biases:  make([]float64, len(layer.Biases)),
		}
		for j := range layer.Weights {
			newLayer.Weights[j] = append([]float64(nil), layer.Weights[j]...)
		}
		copy(newLayer.Biases, layer.Biases)
		clone[i] = newLayer
	}
	return clone
}

func restoreNetwork(layers []*nn.DenseLayer, saved []*nn.DenseLayer) {
	for i := range layers {
		for j := range layers[i].Weights {
			copy(layers[i].Weights[j], saved[i].Weights[j])
		}
		copy(layers[i].Biases, saved[i].Biases)
	}
}
