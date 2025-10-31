package nn

import (
	"fmt"
	"math"
)

// LossFn represents a collection of loss functions.
type LossFn struct{}

// CategoricalCrossEntropy computes the mean categorical cross-entropy loss.
//
// Arguments:
//   - predictions: [][]float64 (softmax outputs, probabilities for each class)
//   - yTrue: []int (true class indices, e.g. [0, 2, 1, ...])
//
// Formula:
//
//	L = - (1/N) * Σ log(p[class_true])
func (lf *LossFn) CategoricalCrossEntropy(predictions [][]float64, yTrue []int) (float64, error) {
	if len(predictions) == 0 {
		return 0, fmt.Errorf("predictions cannot be empty")
	}
	if len(predictions) != len(yTrue) {
		return 0, fmt.Errorf("predictions and labels must have the same length")
	}

	epsilon := 1e-15 // small value to prevent log(0)
	var sumLoss float64

	for i := range predictions {
		classIdx := yTrue[i]
		if classIdx < 0 || classIdx >= len(predictions[i]) {
			return 0, fmt.Errorf("invalid class index %d at sample %d", classIdx, i)
		}

		p := predictions[i][classIdx]
		if p < epsilon {
			p = epsilon
		}
		sumLoss += -math.Log(p)
	}

	meanLoss := sumLoss / float64(len(predictions))
	return meanLoss, nil
}

func (lf *LossFn) SoftmaxCrossEntropyBackward(predictions [][]float64, yTrue []int) [][]float64 {
	samples := len(predictions)
	dInputs := make([][]float64, samples)
	for i := 0; i < samples; i++ {
		dInputs[i] = make([]float64, len(predictions[i]))
		copy(dInputs[i], predictions[i])
		dInputs[i][yTrue[i]] -= 1.0
		for j := range dInputs[i] {
			dInputs[i][j] /= float64(samples)
		}
	}
	return dInputs
}
