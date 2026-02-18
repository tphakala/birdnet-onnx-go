package birdnet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDetectFromShapes(t *testing.T) {
	tests := []struct {
		name    string
		inputs  []tensorInfo
		outputs []tensorInfo
		want    ModelConfig
		wantErr error
	}{
		{
			name:    "BirdNET v2.4",
			inputs:  []tensorInfo{{Name: "input", Dimensions: []int64{1, 144000}}},
			outputs: []tensorInfo{{Name: "output", Dimensions: []int64{1, 6522}}},
			want: ModelConfig{
				ModelType: ModelTypeBirdNetV24, SampleRate: sampleRate48k, Duration: duration3s,
				SampleCount: SampleCountV24, NumSpecies: 6522,
			},
		},
		{
			name:   "BirdNET v3.0",
			inputs: []tensorInfo{{Name: "input", Dimensions: []int64{1, 160000}}},
			outputs: []tensorInfo{
				{Name: "embeddings", Dimensions: []int64{1, 1280}},
				{Name: "logits", Dimensions: []int64{1, 6522}},
			},
			want: ModelConfig{
				ModelType: ModelTypeBirdNetV30, SampleRate: sampleRate32k, Duration: duration5s,
				SampleCount: SampleCountV30, NumSpecies: 6522, EmbeddingDim: 1280,
			},
		},
		{
			name:   "Perch v2",
			inputs: []tensorInfo{{Name: "input", Dimensions: []int64{1, 160000}}},
			outputs: []tensorInfo{
				{Name: "emb", Dimensions: []int64{1, 1536}},
				{Name: "logits", Dimensions: []int64{1, 10000}},
				{Name: "out3", Dimensions: []int64{1, 100}},
				{Name: "out4", Dimensions: []int64{1, 50}},
			},
			want: ModelConfig{
				ModelType: ModelTypePerchV2, SampleRate: sampleRate32k, Duration: duration5s,
				SampleCount: SampleCountPerch, NumSpecies: 10000, EmbeddingDim: 1536,
			},
		},
		{
			name:   "dynamic batch v3.0",
			inputs: []tensorInfo{{Name: "input", Dimensions: []int64{-1, 160000}}},
			outputs: []tensorInfo{
				{Name: "embeddings", Dimensions: []int64{-1, 1280}},
				{Name: "logits", Dimensions: []int64{-1, 6522}},
			},
			want: ModelConfig{
				ModelType: ModelTypeBirdNetV30, SampleRate: sampleRate32k, Duration: duration5s,
				SampleCount: SampleCountV30, NumSpecies: 6522, EmbeddingDim: 1280,
			},
		},
		{
			name:    "dynamic input fallback v2.4",
			inputs:  []tensorInfo{{Name: "input", Dimensions: []int64{1, -1}}},
			outputs: []tensorInfo{{Name: "output", Dimensions: []int64{1, 6522}}},
			want: ModelConfig{
				ModelType: ModelTypeBirdNetV24, SampleRate: sampleRate48k, Duration: duration3s,
				SampleCount: SampleCountV24, NumSpecies: 6522,
			},
		},
		{
			name:   "dynamic input fallback Perch v2",
			inputs: []tensorInfo{{Name: "input", Dimensions: []int64{1, -1}}},
			outputs: []tensorInfo{
				{Name: "emb", Dimensions: []int64{1, 1536}},
				{Name: "logits", Dimensions: []int64{1, 10000}},
				{Name: "out3", Dimensions: []int64{1, 100}},
				{Name: "out4", Dimensions: []int64{1, 50}},
			},
			want: ModelConfig{
				ModelType: ModelTypePerchV2, SampleRate: sampleRate32k, Duration: duration5s,
				SampleCount: SampleCountPerch, NumSpecies: 10000, EmbeddingDim: 1536,
			},
		},
		{
			name:    "unrecognizable",
			inputs:  []tensorInfo{{Name: "input", Dimensions: []int64{1, 99999}}},
			outputs: []tensorInfo{{Name: "output", Dimensions: []int64{1, 100}}},
			wantErr: ErrModelDetection,
		},
		{
			name:    "no inputs",
			inputs:  []tensorInfo{},
			outputs: []tensorInfo{{Name: "output", Dimensions: []int64{1, 100}}},
			wantErr: ErrModelDetection,
		},
		{
			name:    "no outputs",
			inputs:  []tensorInfo{{Name: "input", Dimensions: []int64{1, 144000}}},
			outputs: []tensorInfo{},
			wantErr: ErrModelDetection,
		},
		{
			name:    "empty input dimensions",
			inputs:  []tensorInfo{{Name: "input", Dimensions: []int64{}}},
			outputs: []tensorInfo{{Name: "output", Dimensions: []int64{1, 6522}}},
			wantErr: ErrModelDetection,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := detectFromShapes(tt.inputs, tt.outputs)

			if tt.wantErr != nil {
				require.ErrorIs(t, err, tt.wantErr)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

// --- Task 6: Detection helper tests ---

func TestLastDim(t *testing.T) {
	tests := []struct {
		name string
		dims []int64
		want int
	}{
		{"normal 2D", []int64{1, 6522}, 6522},
		{"3D tensor", []int64{1, 1280, 100}, 100},
		{"1D tensor", []int64{6522}, 6522},
		{"empty dimensions", []int64{}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ti := tensorInfo{Name: "test", Dimensions: tt.dims}
			assert.Equal(t, tt.want, lastDim(ti))
		})
	}
}

func TestConfigBuilders(t *testing.T) {
	t.Run("configV24", func(t *testing.T) {
		outputs := []tensorInfo{{Name: "output", Dimensions: []int64{1, 6522}}}
		cfg := configV24(outputs)
		assert.Equal(t, ModelTypeBirdNetV24, cfg.ModelType)
		assert.Equal(t, sampleRate48k, cfg.SampleRate)
		assert.InDelta(t, duration3s, cfg.Duration, 1e-9)
		assert.Equal(t, SampleCountV24, cfg.SampleCount)
		assert.Equal(t, 6522, cfg.NumSpecies)
	})

	t.Run("configV30", func(t *testing.T) {
		outputs := []tensorInfo{
			{Name: "embeddings", Dimensions: []int64{1, 1280}},
			{Name: "logits", Dimensions: []int64{1, 6522}},
		}
		cfg := configV30(outputs)
		assert.Equal(t, ModelTypeBirdNetV30, cfg.ModelType)
		assert.Equal(t, sampleRate32k, cfg.SampleRate)
		assert.InDelta(t, duration5s, cfg.Duration, 1e-9)
		assert.Equal(t, SampleCountV30, cfg.SampleCount)
		assert.Equal(t, 6522, cfg.NumSpecies)
		assert.Equal(t, 1280, cfg.EmbeddingDim)
	})

	t.Run("configPerch", func(t *testing.T) {
		outputs := []tensorInfo{
			{Name: "emb", Dimensions: []int64{1, 1536}},
			{Name: "logits", Dimensions: []int64{1, 10000}},
		}
		cfg := configPerch(outputs)
		assert.Equal(t, ModelTypePerchV2, cfg.ModelType)
		assert.Equal(t, sampleRate32k, cfg.SampleRate)
		assert.InDelta(t, duration5s, cfg.Duration, 1e-9)
		assert.Equal(t, SampleCountPerch, cfg.SampleCount)
		assert.Equal(t, 10000, cfg.NumSpecies)
		assert.Equal(t, 1536, cfg.EmbeddingDim)
	})
}

func TestDynamicBatchSupported(t *testing.T) {
	tests := []struct {
		name   string
		inputs []tensorInfo
		want   bool
	}{
		{"fixed batch 1", []tensorInfo{{Name: "input", Dimensions: []int64{1, 144000}}}, false},
		{"dynamic batch -1", []tensorInfo{{Name: "input", Dimensions: []int64{-1, 144000}}}, true},
		{"batch greater than 1", []tensorInfo{{Name: "input", Dimensions: []int64{32, 144000}}}, true},
		{"empty inputs", []tensorInfo{}, false},
		{"empty dimensions", []tensorInfo{{Name: "input", Dimensions: []int64{}}}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, dynamicBatchSupported(tt.inputs))
		})
	}
}
