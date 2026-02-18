package birdnet

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClassifierMissingModelPath(t *testing.T) {
	_, err := NewClassifier(
		WithLabels([]string{"sp1", "sp2"}),
	)
	assert.ErrorIs(t, err, ErrModelPath)
}

func TestNewClassifierMissingLabels(t *testing.T) {
	_, err := NewClassifier(
		WithModelPath("nonexistent.onnx"),
	)
	assert.ErrorIs(t, err, ErrLabelsRequired)
}

func TestNewClassifierEmptyModelPath(t *testing.T) {
	_, err := NewClassifier(
		WithModelPath(""),
		WithLabels([]string{"sp1"}),
	)
	assert.Error(t, err, "expected error for empty model path")
}

func TestNewClassifierNoOptions(t *testing.T) {
	_, err := NewClassifier()
	assert.ErrorIs(t, err, ErrModelPath)
}

func TestNewClassifierModelPathOnlyNoLabels(t *testing.T) {
	// Model path is set but no labels source is provided.
	_, err := NewClassifier(
		WithModelPath("some_model.onnx"),
	)
	assert.ErrorIs(t, err, ErrLabelsRequired)
}

// --- Task 3: resolveLabels tests ---

func TestResolveLabels(t *testing.T) {
	t.Run("from direct slice", func(t *testing.T) {
		cfg := &classifierConfig{labels: []string{"sp1", "sp2"}}
		labels, err := resolveLabels(cfg)
		require.NoError(t, err)
		assert.Equal(t, []string{"sp1", "sp2"}, labels)
	})

	t.Run("from file", func(t *testing.T) {
		cfg := &classifierConfig{labelsPath: "testdata/labels_simple.txt"}
		labels, err := resolveLabels(cfg)
		require.NoError(t, err)
		assert.Len(t, labels, 3)
		assert.Equal(t, "Turdus merula_Common Blackbird", labels[0])
	})

	t.Run("from reader", func(t *testing.T) {
		cfg := &classifierConfig{
			labelsReader: strings.NewReader("A\nB\nC\n"),
			labelsFormat: FormatText,
		}
		labels, err := resolveLabels(cfg)
		require.NoError(t, err)
		assert.Equal(t, []string{"A", "B", "C"}, labels)
	})

	t.Run("no source", func(t *testing.T) {
		cfg := &classifierConfig{}
		_, err := resolveLabels(cfg)
		assert.ErrorIs(t, err, ErrLabelsRequired)
	})

	t.Run("priority: direct over file", func(t *testing.T) {
		cfg := &classifierConfig{
			labels:     []string{"direct1", "direct2"},
			labelsPath: "testdata/labels_simple.txt",
		}
		labels, err := resolveLabels(cfg)
		require.NoError(t, err)
		assert.Equal(t, []string{"direct1", "direct2"}, labels)
	})
}

func TestApplyModelTypeOverride(t *testing.T) {
	t.Run("nil override", func(t *testing.T) {
		cfg := ModelConfig{ModelType: ModelTypeBirdNetV24, SampleRate: sampleRate48k}
		applyModelTypeOverride(&cfg, nil)
		assert.Equal(t, ModelTypeBirdNetV24, cfg.ModelType)
		assert.Equal(t, sampleRate48k, cfg.SampleRate)
	})

	t.Run("override to BSG", func(t *testing.T) {
		cfg := ModelConfig{NumSpecies: 6522}
		mt := ModelTypeBSGFinland
		applyModelTypeOverride(&cfg, &mt)
		assert.Equal(t, ModelTypeBSGFinland, cfg.ModelType)
		assert.True(t, cfg.PreSigmoided)
		assert.Equal(t, sampleRate48k, cfg.SampleRate)
		assert.InDelta(t, duration3s, cfg.Duration, 1e-9)
		assert.Equal(t, SampleCountV24, cfg.SampleCount)
		assert.Equal(t, 6522, cfg.NumSpecies, "NumSpecies should be preserved")
	})

	t.Run("override to v3.0", func(t *testing.T) {
		cfg := ModelConfig{NumSpecies: 6522, EmbeddingDim: 1280}
		mt := ModelTypeBirdNetV30
		applyModelTypeOverride(&cfg, &mt)
		assert.Equal(t, ModelTypeBirdNetV30, cfg.ModelType)
		assert.False(t, cfg.PreSigmoided)
		assert.Equal(t, sampleRate32k, cfg.SampleRate)
		assert.InDelta(t, duration5s, cfg.Duration, 1e-9)
		assert.Equal(t, SampleCountV30, cfg.SampleCount)
	})

	t.Run("override to Perch", func(t *testing.T) {
		cfg := ModelConfig{NumSpecies: 10000, EmbeddingDim: 1536}
		mt := ModelTypePerchV2
		applyModelTypeOverride(&cfg, &mt)
		assert.Equal(t, ModelTypePerchV2, cfg.ModelType)
		assert.False(t, cfg.PreSigmoided)
		assert.Equal(t, sampleRate32k, cfg.SampleRate)
		assert.InDelta(t, duration5s, cfg.Duration, 1e-9)
		assert.Equal(t, SampleCountPerch, cfg.SampleCount)
	})

	t.Run("override preserves NumSpecies and EmbeddingDim", func(t *testing.T) {
		cfg := ModelConfig{NumSpecies: 999, EmbeddingDim: 512}
		mt := ModelTypeBirdNetV30
		applyModelTypeOverride(&cfg, &mt)
		assert.Equal(t, 999, cfg.NumSpecies)
		assert.Equal(t, 512, cfg.EmbeddingDim)
	})
}

// --- Task 7: Classifier getter tests ---

func TestClassifierLabelsDefensiveCopy(t *testing.T) {
	c := &Classifier{labels: []string{"sp1", "sp2", "sp3"}}

	copy1 := c.Labels()
	copy1[0] = "mutated"

	copy2 := c.Labels()
	assert.Equal(t, "sp1", copy2[0], "original should be unchanged after mutation")
}

func TestClassifierConfigGetter(t *testing.T) {
	expected := ModelConfig{
		ModelType:   ModelTypeBirdNetV30,
		SampleRate:  sampleRate32k,
		Duration:    duration5s,
		SampleCount: SampleCountV30,
		NumSpecies:  6522,
	}
	c := &Classifier{config: expected}
	assert.Equal(t, expected, c.Config())
}
