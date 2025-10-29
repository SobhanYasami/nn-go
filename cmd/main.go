package main

import (
	"fmt"
	"time"

	dataset "github.com/SobhanYasami/nn-go/internal/data"
	"github.com/SobhanYasami/nn-go/internal/nn"
	"github.com/SobhanYasami/nn-go/internal/utils"
	"github.com/SobhanYasami/nn-go/pkg/logger"
)

func main() {
	log := logger.New("main", logger.DEBUG)
	log.Info("Starting multi-layer neural network demo...")

	start := time.Now() // 🕒 Total runtime timer

	// --- Step 1: Create dataset ---
	stepStart := time.Now()
	X, y := dataset.CreateData(1000, 3)
	fmt.Println("Generated", len(X), "points")

	if err := utils.PlotData(X, y, 3, "spiral.png"); err != nil {
		panic(err)
	}
	fmt.Printf("✅ Dataset & plot done in %v\n", time.Since(stepStart))

	// --- Step 2: Initialize ActivationFn ---
	af := nn.ActivationFn{}

	// --- Step 3: Define layers ---
	layerSizes := []int{2, 8, 8, 6, 6, 4, 3} // 5 hidden layers

	layers := make([]*nn.DenseLayer, len(layerSizes)-1)
	for i := 0; i < len(layerSizes)-1; i++ {
		stepStart = time.Now()
		layer, err := nn.NewDenseLayer(layerSizes[i], layerSizes[i+1])
		if err != nil {
			log.Error("%s", fmt.Sprintf("Error creating layer %d: %v", i+1, err))
			return
		}
		layers[i] = layer
		fmt.Printf("✅ Layer %d (%d→%d) created in %v\n",
			i+1, layerSizes[i], layerSizes[i+1], time.Since(stepStart))
	}

	// --- Step 4: Forward Pass ---
	input := X
	for i, layer := range layers {
		stepStart = time.Now()

		output, _ := layer.Forward(input)
		_ = af.ReLUInPlace(output)
		fmt.Printf("✅ Forward + ReLU (Layer %d) done in %v\n", i+1, time.Since(stepStart))

		input = output
	}

	// --- Step 5: Apply final Softmax ---
	stepStart = time.Now()
	_ = af.SoftmaxInPlace(input)
	fmt.Printf("✅ Final Softmax activation done in %v\n", time.Since(stepStart))

	// --- Step 6: Compute Loss ---
	stepStart = time.Now()
	lf := nn.LossFn{}
	loss, err := lf.CategoricalCrossEntropy(input, y)
	if err != nil {
		log.Error("Error computing loss: %v", err)
		return
	}
	fmt.Printf("✅ Categorical Cross-Entropy Loss: %.6f (computed in %v)\n",
		loss, time.Since(stepStart))

	// --- Step 7: Show sample outputs ---
	fmt.Println("\n🔹 Showing first 10 softmax outputs:")
	for i, sm := range input {
		if i >= 10 {
			break
		}
		fmt.Printf("[%d] %v (label=%d)\n", i, sm, y[i])
	}

	// --- Done ---
	fmt.Printf("\n🚀 Total runtime: %v\n", time.Since(start))
}
