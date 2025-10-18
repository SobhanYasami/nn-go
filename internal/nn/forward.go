package nn

func (l *Layer) Forward(inputs [][]float64) [][]float64 {
	outputs := make([][]float64, len(inputs))
	for i, sample := range inputs {
		out := make([]float64, len(l.Weights))
		for j, neuronWeights := range l.Weights {
			sum := l.Biases[j]
			for k, w := range neuronWeights {
				sum += w * sample[k]
			}
			out[j] = sum
		}
		outputs[i] = out
	}
	return outputs
}
