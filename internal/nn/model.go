package nn

type Model struct {
	Layers []*Layer
}

func NewModel() *Model {
	return &Model{Layers: []*Layer{}}
}

func (m *Model) AddLayer(inputSize, outputSize int) {
	layer := NewLayer(inputSize, outputSize)
	m.Layers = append(m.Layers, layer)
}

func (m *Model) Forward(inputs [][]float64) [][]float64 {
	outputs := inputs
	for _, layer := range m.Layers {
		outputs = layer.Forward(outputs)
	}
	return outputs
}
