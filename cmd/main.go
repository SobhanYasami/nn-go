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

	// --- Step 3: Define layers (input:2 → output:3 → 5 hidden layers → softmax output) ---
	layerSizes := []int{2, 8, 8, 6, 6, 4, 3} // 5 hidden layers in total

	layers := make([]*nn.DenseLayer, len(layerSizes)-1)
	for i := 0; i < len(layerSizes)-1; i++ {
		stepStart = time.Now()
		layer, err := nn.NewDenseLayer(layerSizes[i], layerSizes[i+1])
		if err != nil {
			log.Error(fmt.Sprintf("Error creating layer %d: %v", i+1, err))
			return
		}
		layers[i] = layer
		fmt.Printf("✅ Layer %d (%d→%d) created in %v\n",
			i+1, layerSizes[i], layerSizes[i+1], time.Since(stepStart))
	}

	// --- Step 4: Forward Pass through all layers ---
	input := X
	for i, layer := range layers {
		stepStart = time.Now()

		output, _ := layer.Forward(input)
		_ = af.ReLUInPlace(output) // ReLU after each hidden layer
		fmt.Printf("✅ Forward + ReLU (Layer %d) done in %v\n", i+1, time.Since(stepStart))

		input = output
	}

	// --- Step 5: Apply final Softmax activation ---
	stepStart = time.Now()
	_ = af.SoftmaxInPlace(input)
	fmt.Printf("✅ Final Softmax activation done in %v\n", time.Since(stepStart))

	// --- Step 6: Print sample outputs ---
	fmt.Println("\n🔹 Showing first 10 softmax outputs:")
	for i, sm := range input {
		if i >= 10 {
			break
		}
		fmt.Printf("[%d] %v\n", i, sm)
	}

	// --- Done ---
	fmt.Printf("\n🚀 Total runtime: %v\n", time.Since(start))
}
