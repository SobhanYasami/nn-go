package utils

import (
	"image/color"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// PlotData creates a scatter plot of the generated data.
func PlotData(X [][]float64, y []int, classes int, filename string) error {
	p := plot.New()
	p.Title.Text = "Spiral Dataset"
	p.X.Label.Text = "X1"
	p.Y.Label.Text = "X2"

	// Define colors for each class
	colors := []color.RGBA{
		{R: 255, G: 99, B: 71, A: 255},  // Tomato
		{R: 65, G: 105, B: 225, A: 255}, // Royal Blue
		{R: 34, G: 139, B: 34, A: 255},  // Forest Green
		{R: 255, G: 165, B: 0, A: 255},  // Orange
		{R: 138, G: 43, B: 226, A: 255}, // Blue Violet
	}

	// Plot each class separately for color grouping
	for classNum := 0; classNum < classes; classNum++ {
		pts := make(plotter.XYs, 0)
		for i := range X {
			if y[i] == classNum {
				pts = append(pts, plotter.XY{X: X[i][0], Y: X[i][1]})
			}
		}
		s, err := plotter.NewScatter(pts)
		if err != nil {
			return err
		}
		s.GlyphStyle.Color = colors[classNum%len(colors)]
		s.GlyphStyle.Radius = vg.Points(2.5)
		p.Add(s)
	}

	// Save to PNG file
	if err := p.Save(6*vg.Inch, 6*vg.Inch, filename); err != nil {
		return err
	}
	return nil
}
