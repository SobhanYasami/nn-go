package dataset

import (
	"math"
	"math/rand"
	"time"
)

// CreateData generates a 2D spiral dataset similar to Karpathy's CS231n example.
// samples: number of points per class
// classes: number of classes (spirals)
// returns: X (features), y (labels)
func CreateData(samples, classes int) ([][]float64, []int) {
	// Local random generator (no deprecated global seed)
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	total := samples * classes
	X := make([][]float64, total)
	y := make([]int, total)

	for classNum := 0; classNum < classes; classNum++ {
		for i := 0; i < samples; i++ {
			ix := i + classNum*samples

			r := float64(i) / float64(samples-1) // radius
			t := float64(classNum)*4.0 + float64(i)/float64(samples-1)*4.0
			t += rng.NormFloat64() * 0.2 // add gaussian noise

			X[ix] = []float64{
				r * math.Sin(t*2.5),
				r * math.Cos(t*2.5),
			}
			y[ix] = classNum
		}
	}
	return X, y
}
