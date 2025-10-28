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

	log.Info("Starting mini neural network demo...")
	start := time.Now() // 🕒 Start total runtime timer

	// --- Step 1: Create dataset ---
	stepStart := time.Now()
	X, y := dataset.CreateData(1000, 3)
	fmt.Println("Generated", len(X), "points")

	if err := utils.PlotData(X, y, 3, "spiral.png"); err != nil {
		panic(err)
	}
	fmt.Printf("✅ Dataset & plot done in %v\n", time.Since(stepStart))

	// --- Step 2: Create Dense Layer ---
	stepStart = time.Now()
	layer_1, err := nn.NewDenseLayer(2, 3)
	if err != nil {
		fmt.Println("Error creating new layer:", err)
	}
	fmt.Printf("✅ Layer creation done in %v\n", time.Since(stepStart))

	// --- Step 3: Forward Pass ---
	stepStart = time.Now()
	dl1Output, _ := layer_1.Forward(X)
	fmt.Printf("✅ Forward pass done in %v\n", time.Since(stepStart))

	// fmt.Println("===================")
	// fmt.Println("output of layer1:", dl1Output)
	// fmt.Println("===================")

	// --- Step 4: Activation (ReLU) ---
	stepStart = time.Now()
	af := nn.ActivationFn{}
	_ = af.ReLUInPlace(dl1Output)
	fmt.Printf("✅ ReLU activation done in %v\n", time.Since(stepStart))

	// fmt.Println("In-place ReLU output:", dl1Output)

	// --- Done ---
	fmt.Printf("\n🚀 Total runtime: %v\n", time.Since(start))

}
