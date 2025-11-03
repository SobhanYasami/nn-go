package nn

import (
	"fmt"
	"math"
	"sync"
)

// ActivationType represents different activation functions
type ActivationType string

const (
	ReLU      ActivationType = "relu"
	Sigmoid   ActivationType = "sigmoid"
	Tanh      ActivationType = "tanh"
	Softmax   ActivationType = "softmax"
	Linear    ActivationType = "linear"
	LeakyReLU ActivationType = "leaky_relu"
	ELU       ActivationType = "elu"
)

// ActivationFn provides activation functions and their derivatives
type ActivationFn struct {
	// Cache for storing intermediate values during forward pass for efficient backward pass
	cache sync.Map
}

// ActivationResult holds both the output and a function to compute gradients
type ActivationResult struct {
	Output   [][]float64
	Backward func(dOutputs [][]float64) [][]float64
}

// NewActivationFn creates a new ActivationFn instance
func NewActivationFn() *ActivationFn {
	return &ActivationFn{}
}

//? ------------------------------
//? Interface Methods
//? ------------------------------

// Apply applies the specified activation function to inputs
func (af *ActivationFn) Apply(activation ActivationType, inputs [][]float64, inPlace bool) ([][]float64, error) {
	switch activation {
	case ReLU:
		if inPlace {
			return nil, af.ReLUInPlace(inputs)
		}
		return af.ReLU(inputs)
	case Sigmoid:
		if inPlace {
			return nil, af.SigmoidInPlace(inputs)
		}
		return af.Sigmoid(inputs)
	case Tanh:
		if inPlace {
			return nil, af.TanhInPlace(inputs)
		}
		return af.Tanh(inputs)
	case Softmax:
		if inPlace {
			return nil, af.SoftmaxInPlace(inputs)
		}
		return af.Softmax(inputs)
	case LeakyReLU:
		if inPlace {
			return nil, af.LeakyReLUInPlace(inputs, 0.01)
		}
		return af.LeakyReLU(inputs, 0.01)
	case ELU:
		if inPlace {
			return nil, af.ELUInPlace(inputs, 1.0)
		}
		return af.ELU(inputs, 1.0)
	case Linear:
		// Linear activation returns inputs as-is
		if inPlace {
			return nil, nil
		}
		return copyMatrix(inputs), nil
	default:
		return nil, fmt.Errorf("unknown activation function: %s", activation)
	}
}

// ApplyWithGrad applies activation and returns result with gradient function
func (af *ActivationFn) ApplyWithGrad(activation ActivationType, inputs [][]float64) (*ActivationResult, error) {
	var output [][]float64
	var err error

	switch activation {
	case ReLU:
		output, err = af.ReLU(inputs)
		if err != nil {
			return nil, err
		}
		return &ActivationResult{
			Output: output,
			Backward: func(dOutputs [][]float64) [][]float64 {
				return af.ReLUBackward(dOutputs, inputs)
			},
		}, nil

	case Sigmoid:
		output, err = af.Sigmoid(inputs)
		if err != nil {
			return nil, err
		}
		return &ActivationResult{
			Output: output,
			Backward: func(dOutputs [][]float64) [][]float64 {
				return af.SigmoidBackward(dOutputs, output)
			},
		}, nil

	case Tanh:
		output, err = af.Tanh(inputs)
		if err != nil {
			return nil, err
		}
		return &ActivationResult{
			Output: output,
			Backward: func(dOutputs [][]float64) [][]float64 {
				return af.TanhBackward(dOutputs, output)
			},
		}, nil

	case Softmax:
		output, err = af.Softmax(inputs)
		if err != nil {
			return nil, err
		}
		return &ActivationResult{
			Output: output,
			Backward: func(dOutputs [][]float64) [][]float64 {
				return af.SoftmaxBackward(dOutputs, output)
			},
		}, nil

	default:
		return nil, fmt.Errorf("gradient not implemented for activation: %s", activation)
	}
}

//? ------------------------------
//? ReLU Family Activations
//? ------------------------------

// ReLUInPlace applies ReLU(x) = max(0, x) elementwise in-place.
func (af *ActivationFn) ReLUInPlace(inputs [][]float64) error {
	if err := validateMatrix(inputs); err != nil {
		return err
	}

	for i := range inputs {
		for j := range inputs[i] {
			if inputs[i][j] < 0 {
				inputs[i][j] = 0
			}
		}
	}
	return nil
}

// ReLU returns a new slice (non-mutating version).
func (af *ActivationFn) ReLU(inputs [][]float64) ([][]float64, error) {
	if err := validateMatrix(inputs); err != nil {
		return nil, err
	}

	output := make([][]float64, len(inputs))
	for i := range inputs {
		output[i] = make([]float64, len(inputs[i]))
		for j := range inputs[i] {
			if inputs[i][j] > 0 {
				output[i][j] = inputs[i][j]
			}
		}
	}
	return output, nil
}

// LeakyReLU applies Leaky ReLU: f(x) = x if x > 0, else alpha * x
func (af *ActivationFn) LeakyReLU(inputs [][]float64, alpha float64) ([][]float64, error) {
	if err := validateMatrix(inputs); err != nil {
		return nil, err
	}

	output := make([][]float64, len(inputs))
	for i := range inputs {
		output[i] = make([]float64, len(inputs[i]))
		for j := range inputs[i] {
			if inputs[i][j] > 0 {
				output[i][j] = inputs[i][j]
			} else {
				output[i][j] = alpha * inputs[i][j]
			}
		}
	}
	return output, nil
}

// LeakyReLUInPlace applies Leaky ReLU in-place
func (af *ActivationFn) LeakyReLUInPlace(inputs [][]float64, alpha float64) error {
	if err := validateMatrix(inputs); err != nil {
		return err
	}

	for i := range inputs {
		for j := range inputs[i] {
			if inputs[i][j] < 0 {
				inputs[i][j] *= alpha
			}
		}
	}
	return nil
}

// ELU applies Exponential Linear Unit: f(x) = x if x > 0, else alpha * (exp(x) - 1)
func (af *ActivationFn) ELU(inputs [][]float64, alpha float64) ([][]float64, error) {
	if err := validateMatrix(inputs); err != nil {
		return nil, err
	}

	output := make([][]float64, len(inputs))
	for i := range inputs {
		output[i] = make([]float64, len(inputs[i]))
		for j := range inputs[i] {
			if inputs[i][j] > 0 {
				output[i][j] = inputs[i][j]
			} else {
				output[i][j] = alpha * (math.Exp(inputs[i][j]) - 1)
			}
		}
	}
	return output, nil
}

// ELUInPlace applies ELU in-place
func (af *ActivationFn) ELUInPlace(inputs [][]float64, alpha float64) error {
	if err := validateMatrix(inputs); err != nil {
		return err
	}

	for i := range inputs {
		for j := range inputs[i] {
			if inputs[i][j] < 0 {
				inputs[i][j] = alpha * (math.Exp(inputs[i][j]) - 1)
			}
		}
	}
	return nil
}

//? ------------------------------
//? Sigmoid Activations
//? ------------------------------

// Sigmoid applies sigmoid activation: f(x) = 1 / (1 + exp(-x))
func (af *ActivationFn) Sigmoid(inputs [][]float64) ([][]float64, error) {
	if err := validateMatrix(inputs); err != nil {
		return nil, err
	}

	output := make([][]float64, len(inputs))
	for i := range inputs {
		output[i] = make([]float64, len(inputs[i]))
		for j := range inputs[i] {
			output[i][j] = sigmoid(inputs[i][j])
		}
	}
	return output, nil
}

// SigmoidInPlace applies sigmoid activation in-place
func (af *ActivationFn) SigmoidInPlace(inputs [][]float64) error {
	if err := validateMatrix(inputs); err != nil {
		return err
	}

	for i := range inputs {
		for j := range inputs[i] {
			inputs[i][j] = sigmoid(inputs[i][j])
		}
	}
	return nil
}

//? ------------------------------
//? Tanh Activations
//? ------------------------------

// Tanh applies hyperbolic tangent activation: f(x) = tanh(x)
func (af *ActivationFn) Tanh(inputs [][]float64) ([][]float64, error) {
	if err := validateMatrix(inputs); err != nil {
		return nil, err
	}

	output := make([][]float64, len(inputs))
	for i := range inputs {
		output[i] = make([]float64, len(inputs[i]))
		for j := range inputs[i] {
			output[i][j] = math.Tanh(inputs[i][j])
		}
	}
	return output, nil
}

// TanhInPlace applies tanh activation in-place
func (af *ActivationFn) TanhInPlace(inputs [][]float64) error {
	if err := validateMatrix(inputs); err != nil {
		return err
	}

	for i := range inputs {
		for j := range inputs[i] {
			inputs[i][j] = math.Tanh(inputs[i][j])
		}
	}
	return nil
}

//? ------------------------------
//? Softmax Activations
//? ------------------------------

// SoftmaxInPlace applies softmax activation in-place with numerical stability
func (af *ActivationFn) SoftmaxInPlace(inputs [][]float64) error {
	if err := validateMatrix(inputs); err != nil {
		return err
	}

	for i := range inputs {
		if len(inputs[i]) == 0 {
			continue
		}

		maxVal := findMax(inputs[i])
		var sum float64

		// Compute exp(x - max) and sum
		for j := range inputs[i] {
			inputs[i][j] = math.Exp(inputs[i][j] - maxVal)
			sum += inputs[i][j]
		}

		// Normalize
		for j := range inputs[i] {
			inputs[i][j] /= sum
		}
	}
	return nil
}

// Softmax returns a new slice with softmax applied
func (af *ActivationFn) Softmax(inputs [][]float64) ([][]float64, error) {
	if err := validateMatrix(inputs); err != nil {
		return nil, err
	}

	output := make([][]float64, len(inputs))
	for i := range inputs {
		if len(inputs[i]) == 0 {
			output[i] = []float64{}
			continue
		}

		maxVal := findMax(inputs[i])
		output[i] = make([]float64, len(inputs[i]))
		var sum float64

		// Compute exp(x - max) and sum
		for j := range inputs[i] {
			output[i][j] = math.Exp(inputs[i][j] - maxVal)
			sum += output[i][j]
		}

		// Normalize
		for j := range output[i] {
			output[i][j] /= sum
		}
	}
	return output, nil
}

//? ------------------------------
//? Backward Pass Methods
//? ------------------------------

// ReLUBackward computes gradient for ReLU activation
func (af *ActivationFn) ReLUBackward(dOutputs, inputs [][]float64) [][]float64 {
	if err := validateGradients(dOutputs, inputs); err != nil {
		return nil
	}

	dInputs := make([][]float64, len(dOutputs))
	for i := range dOutputs {
		dInputs[i] = make([]float64, len(dOutputs[i]))
		for j := range dOutputs[i] {
			if inputs[i][j] > 0 {
				dInputs[i][j] = dOutputs[i][j]
			}
		}
	}
	return dInputs
}

// LeakyReLUBackward computes gradient for Leaky ReLU activation
func (af *ActivationFn) LeakyReLUBackward(dOutputs, inputs [][]float64, alpha float64) [][]float64 {
	if err := validateGradients(dOutputs, inputs); err != nil {
		return nil
	}

	dInputs := make([][]float64, len(dOutputs))
	for i := range dOutputs {
		dInputs[i] = make([]float64, len(dOutputs[i]))
		for j := range dOutputs[i] {
			if inputs[i][j] > 0 {
				dInputs[i][j] = dOutputs[i][j]
			} else {
				dInputs[i][j] = alpha * dOutputs[i][j]
			}
		}
	}
	return dInputs
}

// SigmoidBackward computes gradient for sigmoid activation
func (af *ActivationFn) SigmoidBackward(dOutputs, outputs [][]float64) [][]float64 {
	if err := validateGradients(dOutputs, outputs); err != nil {
		return nil
	}

	dInputs := make([][]float64, len(dOutputs))
	for i := range dOutputs {
		dInputs[i] = make([]float64, len(dOutputs[i]))
		for j := range dOutputs[i] {
			dInputs[i][j] = dOutputs[i][j] * outputs[i][j] * (1 - outputs[i][j])
		}
	}
	return dInputs
}

// TanhBackward computes gradient for tanh activation
func (af *ActivationFn) TanhBackward(dOutputs, outputs [][]float64) [][]float64 {
	if err := validateGradients(dOutputs, outputs); err != nil {
		return nil
	}

	dInputs := make([][]float64, len(dOutputs))
	for i := range dOutputs {
		dInputs[i] = make([]float64, len(dOutputs[i]))
		for j := range dOutputs[i] {
			dInputs[i][j] = dOutputs[i][j] * (1 - outputs[i][j]*outputs[i][j])
		}
	}
	return dInputs
}

// SoftmaxBackward computes gradient for softmax activation (assumes cross-entropy loss)
func (af *ActivationFn) SoftmaxBackward(dOutputs, outputs [][]float64) [][]float64 {
	if err := validateGradients(dOutputs, outputs); err != nil {
		return nil
	}

	// For softmax + cross-entropy, the gradient is simply (output - target)
	// which is already provided in dOutputs
	return dOutputs
}

//? ------------------------------
//? Utility Functions
//? ------------------------------

// validateMatrix checks if the input matrix is valid
func validateMatrix(matrix [][]float64) error {
	if len(matrix) == 0 {
		return fmt.Errorf("matrix cannot be empty")
	}
	cols := len(matrix[0])
	for i := range matrix {
		if len(matrix[i]) != cols {
			return fmt.Errorf("matrix has inconsistent column lengths")
		}
	}
	return nil
}

// validateGradients checks if gradient matrices are compatible
func validateGradients(dOutputs, inputs [][]float64) error {
	if len(dOutputs) != len(inputs) {
		return fmt.Errorf("gradient and input have different batch sizes")
	}
	if len(dOutputs) > 0 && len(dOutputs[0]) != len(inputs[0]) {
		return fmt.Errorf("gradient and input have different feature dimensions")
	}
	return nil
}

// findMax returns the maximum value in a slice
func findMax(slice []float64) float64 {
	if len(slice) == 0 {
		return 0
	}
	max := slice[0]
	for _, v := range slice[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

// copyMatrix creates a deep copy of a matrix
func copyMatrix(matrix [][]float64) [][]float64 {
	result := make([][]float64, len(matrix))
	for i := range matrix {
		result[i] = make([]float64, len(matrix[i]))
		copy(result[i], matrix[i])
	}
	return result
}

//? ------------------------------
//? Scalar Activation Functions (for reference/internal use)
//? ------------------------------

func sigmoid(x float64) float64 {
	// Clip to prevent overflow
	if x < -20 {
		return 0
	}
	if x > 20 {
		return 1
	}
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func reluDerivative(x float64) float64 {
	if x > 0 {
		return 1.0
	}
	return 0.0
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}

func tanhDerivative(x float64) float64 {
	return 1.0 - math.Pow(math.Tanh(x), 2)
}
