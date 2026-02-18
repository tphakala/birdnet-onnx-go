package birdnet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSigmoid(t *testing.T) {
	tests := []struct {
		name    string
		input   float32
		want    float32
		epsilon float64
	}{
		{"zero", 0, 0.5, 1e-6},
		{"large positive", 100, 1.0, 1e-6},
		{"large negative", -100, 0.0, 1e-6},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Sigmoid(tt.input)
			assert.InDelta(t, tt.want, got, tt.epsilon)
		})
	}
}

func TestTopKPredictions(t *testing.T) {
	scores := []float32{0.1, 5.0, -2.0, 3.5, 0.0, 1.2, -1.0, 4.0, 2.5, -0.5}
	labels := []string{"sp0", "sp1", "sp2", "sp3", "sp4", "sp5", "sp6", "sp7", "sp8", "sp9"}

	results := TopKPredictions(scores, labels, 3, 0.0, false)
	require.Len(t, results, 3)

	// Verify order: sp1 (5.0), sp7 (4.0), sp3 (3.5) after sigmoid.
	assert.Equal(t, "sp1", results[0].Species)
	assert.Equal(t, 1, results[0].Index)
	assert.GreaterOrEqual(t, results[0].Confidence, float32(0.99))

	assert.Equal(t, "sp7", results[1].Species)
	assert.Equal(t, 7, results[1].Index)
	assert.GreaterOrEqual(t, results[1].Confidence, float32(0.98))

	assert.Equal(t, "sp3", results[2].Species)
	assert.Equal(t, 3, results[2].Index)
	assert.GreaterOrEqual(t, results[2].Confidence, float32(0.97))

	// Verify descending order.
	for i := 1; i < len(results); i++ {
		assert.GreaterOrEqual(t, results[i-1].Confidence, results[i].Confidence,
			"results should be sorted descending by confidence")
	}
}

func TestTopKPredictionsWithMinConfidence(t *testing.T) {
	scores := []float32{0.1, 5.0, -2.0, 3.5, 0.0, 1.2, -1.0, 4.0, 2.5, -0.5}
	labels := []string{"sp0", "sp1", "sp2", "sp3", "sp4", "sp5", "sp6", "sp7", "sp8", "sp9"}

	// With minConf=0.99, only sp1 (sigmoid(5.0)≈0.993) should pass.
	results := TopKPredictions(scores, labels, 3, 0.99, false)
	require.Len(t, results, 1)
	assert.Equal(t, "sp1", results[0].Species)
}

func TestTopKPredictionsPreSigmoided(t *testing.T) {
	scores := []float32{0.1, 0.95, 0.3, 0.85, 0.5}
	labels := []string{"sp0", "sp1", "sp2", "sp3", "sp4"}

	results := TopKPredictions(scores, labels, 3, 0.0, true)
	require.Len(t, results, 3)

	assert.Equal(t, "sp1", results[0].Species)
	assert.Equal(t, float32(0.95), results[0].Confidence)

	assert.Equal(t, "sp3", results[1].Species)
	assert.Equal(t, float32(0.85), results[1].Confidence)

	assert.Equal(t, "sp4", results[2].Species)
	assert.Equal(t, float32(0.5), results[2].Confidence)
}

func TestTopKPredictionsEmptyInput(t *testing.T) {
	results := TopKPredictions([]float32{}, []string{}, 3, 0.0, false)
	assert.Empty(t, results)

	results = TopKPredictions(nil, nil, 3, 0.0, false)
	assert.Empty(t, results)
}
