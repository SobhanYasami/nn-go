package nn

import "math"

func RoundTo(x float64, digits int) float64 {
	pow := math.Pow(10, float64(digits))
	return math.Round(x*pow) / pow
}
