package birdnet

import (
	"math"
	"testing"
)

func TestSigmoid(t *testing.T) {
	tests := []struct {
		name    string
		input   float32
		want    float32
		epsilon float32
	}{
		{"zero", 0, 0.5, 1e-6},
		{"large positive", 100, 1.0, 1e-6},
		{"large negative", -100, 0.0, 1e-6},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Sigmoid(tt.input)
			if float32(math.Abs(float64(got-tt.want))) > tt.epsilon {
				t.Errorf("Sigmoid(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestTopKPredictions(t *testing.T) {
	scores := []float32{0.1, 5.0, -2.0, 3.5, 0.0, 1.2, -1.0, 4.0, 2.5, -0.5}
	labels := []string{"sp0", "sp1", "sp2", "sp3", "sp4", "sp5", "sp6", "sp7", "sp8", "sp9"}

	results := TopKPredictions(scores, labels, 3, 0.0, false)

	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	// Verify order: sp1 (5.0), sp7 (4.0), sp3 (3.5) after sigmoid
	expectedOrder := []struct {
		species string
		index   int
		minConf float32
	}{
		{"sp1", 1, 0.99},
		{"sp7", 7, 0.98},
		{"sp3", 3, 0.97},
	}

	for i, exp := range expectedOrder {
		if results[i].Species != exp.species {
			t.Errorf("result[%d].Species = %q, want %q", i, results[i].Species, exp.species)
		}
		if results[i].Index != exp.index {
			t.Errorf("result[%d].Index = %d, want %d", i, results[i].Index, exp.index)
		}
		if results[i].Confidence < exp.minConf {
			t.Errorf("result[%d].Confidence = %v, want >= %v", i, results[i].Confidence, exp.minConf)
		}
	}

	// Verify descending order
	for i := 1; i < len(results); i++ {
		if results[i].Confidence > results[i-1].Confidence {
			t.Errorf("results not sorted descending: [%d]=%v > [%d]=%v",
				i, results[i].Confidence, i-1, results[i-1].Confidence)
		}
	}
}

func TestTopKPredictionsWithMinConfidence(t *testing.T) {
	scores := []float32{0.1, 5.0, -2.0, 3.5, 0.0, 1.2, -1.0, 4.0, 2.5, -0.5}
	labels := []string{"sp0", "sp1", "sp2", "sp3", "sp4", "sp5", "sp6", "sp7", "sp8", "sp9"}

	// With minConf=0.99, only sp1 (sigmoid(5.0)≈0.993) should pass
	results := TopKPredictions(scores, labels, 3, 0.99, false)

	if len(results) != 1 {
		t.Fatalf("expected 1 result with minConf=0.99, got %d: %+v", len(results), results)
	}
	if results[0].Species != "sp1" {
		t.Errorf("expected sp1, got %q", results[0].Species)
	}
}

func TestTopKPredictionsPreSigmoided(t *testing.T) {
	// BSG model: scores are already in [0,1], no sigmoid should be applied
	scores := []float32{0.1, 0.95, 0.3, 0.85, 0.5}
	labels := []string{"sp0", "sp1", "sp2", "sp3", "sp4"}

	results := TopKPredictions(scores, labels, 3, 0.0, true)

	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	// Scores should be used directly without sigmoid
	if results[0].Species != "sp1" || results[0].Confidence != 0.95 {
		t.Errorf("result[0] = %+v, want sp1 with confidence 0.95", results[0])
	}
	if results[1].Species != "sp3" || results[1].Confidence != 0.85 {
		t.Errorf("result[1] = %+v, want sp3 with confidence 0.85", results[1])
	}
	if results[2].Species != "sp4" || results[2].Confidence != 0.5 {
		t.Errorf("result[2] = %+v, want sp4 with confidence 0.5", results[2])
	}
}

func TestTopKPredictionsEmptyInput(t *testing.T) {
	results := TopKPredictions([]float32{}, []string{}, 3, 0.0, false)

	if len(results) != 0 {
		t.Errorf("expected empty result for empty input, got %d results", len(results))
	}

	// Also test nil inputs
	results = TopKPredictions(nil, nil, 3, 0.0, false)
	if len(results) != 0 {
		t.Errorf("expected empty result for nil input, got %d results", len(results))
	}
}
