package mathx

import (
	"math"
	"reflect"
	"testing"
)

func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func TestRoundTo(t *testing.T) {
	ng := &NumGo{}
	tests := []struct {
		x      float64
		digits int
		want   float64
	}{
		{3.14159, 2, 3.14},
		{2.71828, 3, 2.718},
		{1.9999, 0, 2},
		{-1.2345, 2, -1.23},
	}

	for _, tt := range tests {
		got := ng.RoundTo(tt.x, tt.digits)
		if !almostEqual(got, tt.want, 1e-9) {
			t.Errorf("RoundTo(%f, %d) = %f; want %f", tt.x, tt.digits, got, tt.want)
		}
	}
}

func TestDotVectors(t *testing.T) {
	ng := &NumGo{}
	v1 := []float64{1, 2, 3}
	v2 := []float64{4, 5, 6}

	got, err := ng.DotVectors(v1, v2)
	if err != nil {
		t.Fatalf("DotVectors() returned error: %v", err)
	}
	want := 32.0 // 1*4 + 2*5 + 3*6
	if got != want {
		t.Errorf("DotVectors() = %v; want %v", got, want)
	}

	_, err = ng.DotVectors([]float64{1, 2}, []float64{3})
	if err == nil {
		t.Error("DotVectors() expected error on mismatched lengths, got nil")
	}
}

func TestAddSubScaleVectors(t *testing.T) {
	ng := &NumGo{}
	v1 := []float64{1, 2, 3}
	v2 := []float64{4, 5, 6}

	add, _ := ng.AddVectors(v1, v2)
	wantAdd := []float64{5, 7, 9}
	if !reflect.DeepEqual(add, wantAdd) {
		t.Errorf("AddVectors() = %v; want %v", add, wantAdd)
	}

	sub, _ := ng.SubVectors(v1, v2)
	wantSub := []float64{-3, -3, -3}
	if !reflect.DeepEqual(sub, wantSub) {
		t.Errorf("SubVectors() = %v; want %v", sub, wantSub)
	}

	scaled := ng.ScaleVector(v1, 2)
	wantScaled := []float64{2, 4, 6}
	if !reflect.DeepEqual(scaled, wantScaled) {
		t.Errorf("ScaleVector() = %v; want %v", scaled, wantScaled)
	}
}

func TestNormAndNormalize(t *testing.T) {
	ng := &NumGo{}
	v := []float64{3, 4}

	norm := ng.Norm(v)
	if !almostEqual(norm, 5, 1e-9) {
		t.Errorf("Norm() = %v; want 5", norm)
	}

	unit := ng.Normalize(v)
	want := []float64{0.6, 0.8}
	for i := range unit {
		if !almostEqual(unit[i], want[i], 1e-9) {
			t.Errorf("Normalize() = %v; want %v", unit, want)
			break
		}
	}
}

func TestDotMatrix(t *testing.T) {
	ng := &NumGo{}
	A := [][]float64{
		{1, 2},
		{3, 4},
	}
	B := [][]float64{
		{5, 6},
		{7, 8},
	}
	want := [][]float64{
		{19, 22},
		{43, 50},
	}

	C, err := ng.DotMatrix(A, B)
	if err != nil {
		t.Fatalf("DotMatrix() returned error: %v", err)
	}

	for i := range C {
		for j := range C[i] {
			if !almostEqual(C[i][j], want[i][j], 1e-9) {
				t.Errorf("DotMatrix()[%d][%d] = %v; want %v", i, j, C[i][j], want[i][j])
			}
		}
	}

	// Invalid dimension test
	_, err = ng.DotMatrix([][]float64{{1, 2}}, [][]float64{{3}, {4}, {5}})
	if err == nil {
		t.Error("DotMatrix() expected dimension mismatch error, got nil")
	}
}

func TestTranspose(t *testing.T) {
	ng := &NumGo{}
	M := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}
	want := [][]float64{
		{1, 4},
		{2, 5},
		{3, 6},
	}

	got := ng.Transpose(M)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Transpose() = %v; want %v", got, want)
	}
}

func TestNumGo_RoundTo(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for target function.
		x      float64
		digits int
		want   float64
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// TODO: construct the receiver type.
			var ng NumGo
			got := ng.RoundTo(tt.x, tt.digits)
			// TODO: update the condition below to compare got with tt.want.
			if true {
				t.Errorf("RoundTo() = %v, want %v", got, tt.want)
			}
		})
	}
}
