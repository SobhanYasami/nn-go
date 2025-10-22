package nn

type Model struct {
	Layers []*DenseLayer
}

func NewModel() *Model {
	return &Model{Layers: []*DenseLayer{}}
}

func (m *Model) AddLayer(inputSize, outputSize int) {
	layer, _ := NewDenseLayer(inputSize, outputSize)
	m.Layers = append(m.Layers, layer)
}

func (m *Model) Forward(inputs [][]float64) [][]float64 {
	outputs := inputs
	for _, layer := range m.Layers {
		outputs, _ = layer.Forward(outputs)
	}
	return outputs
}
