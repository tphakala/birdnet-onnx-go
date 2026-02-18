package birdnet

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWithModelPath(t *testing.T) {
	t.Run("valid path", func(t *testing.T) {
		cfg := defaultConfig()
		err := WithModelPath("/some/model.onnx")(&cfg)
		require.NoError(t, err)
		assert.Equal(t, "/some/model.onnx", cfg.modelPath)
	})

	t.Run("empty path", func(t *testing.T) {
		cfg := defaultConfig()
		err := WithModelPath("")(&cfg)
		assert.Error(t, err)
	})
}

func TestWithLabelsFromFile(t *testing.T) {
	t.Run("valid path", func(t *testing.T) {
		cfg := defaultConfig()
		err := WithLabelsFromFile("/labels.txt")(&cfg)
		require.NoError(t, err)
		assert.Equal(t, "/labels.txt", cfg.labelsPath)
	})

	t.Run("empty path", func(t *testing.T) {
		cfg := defaultConfig()
		err := WithLabelsFromFile("")(&cfg)
		assert.Error(t, err)
	})
}

func TestWithLabelsFromReader(t *testing.T) {
	t.Run("valid reader and format", func(t *testing.T) {
		cfg := defaultConfig()
		r := strings.NewReader("sp1\nsp2\n")
		err := WithLabelsFromReader(r, FormatText)(&cfg)
		require.NoError(t, err)
		assert.Equal(t, r, cfg.labelsReader)
		assert.Equal(t, FormatText, cfg.labelsFormat)
	})

	t.Run("nil reader", func(t *testing.T) {
		cfg := defaultConfig()
		err := WithLabelsFromReader(nil, FormatText)(&cfg)
		assert.Error(t, err)
	})

	t.Run("bad format", func(t *testing.T) {
		cfg := defaultConfig()
		r := strings.NewReader("data")
		err := WithLabelsFromReader(r, "xml")(&cfg)
		assert.Error(t, err)
	})

	t.Run("csv format", func(t *testing.T) {
		cfg := defaultConfig()
		r := strings.NewReader("data")
		err := WithLabelsFromReader(r, FormatCSV)(&cfg)
		require.NoError(t, err)
		assert.Equal(t, FormatCSV, cfg.labelsFormat)
	})

	t.Run("json format", func(t *testing.T) {
		cfg := defaultConfig()
		r := strings.NewReader("data")
		err := WithLabelsFromReader(r, FormatJSON)(&cfg)
		require.NoError(t, err)
		assert.Equal(t, "json", cfg.labelsFormat)
	})
}

func TestWithLabels(t *testing.T) {
	t.Run("valid slice", func(t *testing.T) {
		cfg := defaultConfig()
		err := WithLabels([]string{"sp1", "sp2"})(&cfg)
		require.NoError(t, err)
		assert.Equal(t, []string{"sp1", "sp2"}, cfg.labels)
	})

	t.Run("empty slice", func(t *testing.T) {
		cfg := defaultConfig()
		err := WithLabels([]string{})(&cfg)
		assert.Error(t, err)
	})

	t.Run("nil slice", func(t *testing.T) {
		cfg := defaultConfig()
		err := WithLabels(nil)(&cfg)
		assert.Error(t, err)
	})
}

func TestWithModelType(t *testing.T) {
	types := []ModelType{ModelTypeBirdNetV24, ModelTypeBirdNetV30, ModelTypePerchV2, ModelTypeBSGFinland}
	for _, mt := range types {
		t.Run(mt.String(), func(t *testing.T) {
			cfg := defaultConfig()
			err := WithModelType(mt)(&cfg)
			require.NoError(t, err)
			require.NotNil(t, cfg.modelType)
			assert.Equal(t, mt, *cfg.modelType)
		})
	}
}

func TestWithTopK(t *testing.T) {
	t.Run("positive", func(t *testing.T) {
		cfg := defaultConfig()
		err := WithTopK(5)(&cfg)
		require.NoError(t, err)
		assert.Equal(t, 5, cfg.topK)
	})

	t.Run("zero", func(t *testing.T) {
		cfg := defaultConfig()
		err := WithTopK(0)(&cfg)
		assert.Error(t, err)
	})

	t.Run("negative", func(t *testing.T) {
		cfg := defaultConfig()
		err := WithTopK(-1)(&cfg)
		assert.Error(t, err)
	})
}

func TestWithMinConfidence(t *testing.T) {
	t.Run("zero", func(t *testing.T) {
		cfg := defaultConfig()
		err := WithMinConfidence(0.0)(&cfg)
		require.NoError(t, err)
		assert.InDelta(t, float32(0.0), cfg.minConf, 1e-9)
	})

	t.Run("valid", func(t *testing.T) {
		cfg := defaultConfig()
		err := WithMinConfidence(0.5)(&cfg)
		require.NoError(t, err)
		assert.InDelta(t, float32(0.5), cfg.minConf, 1e-9)
	})

	t.Run("negative", func(t *testing.T) {
		cfg := defaultConfig()
		err := WithMinConfidence(-0.1)(&cfg)
		assert.Error(t, err)
	})
}

func TestWithSessionOptions(t *testing.T) {
	t.Run("nil", func(t *testing.T) {
		cfg := defaultConfig()
		err := WithSessionOptions(nil)(&cfg)
		assert.Error(t, err)
	})
}

func TestDefaultConfig(t *testing.T) {
	cfg := defaultConfig()
	assert.Equal(t, defaultTopK, cfg.topK)
	assert.InDelta(t, float32(0.0), cfg.minConf, 1e-9)
	assert.Empty(t, cfg.modelPath)
	assert.Nil(t, cfg.labels)
	assert.Nil(t, cfg.modelType)
}
