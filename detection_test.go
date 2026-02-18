package birdnet

import (
	"errors"
	"testing"
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
				ModelType: ModelTypeBirdNetV24, SampleRate: 48000, Duration: 3.0,
				SampleCount: 144000, NumSpecies: 6522, EmbeddingDim: 0, PreSigmoided: false,
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
				ModelType: ModelTypeBirdNetV30, SampleRate: 32000, Duration: 5.0,
				SampleCount: 160000, NumSpecies: 6522, EmbeddingDim: 1280, PreSigmoided: false,
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
				ModelType: ModelTypePerchV2, SampleRate: 32000, Duration: 5.0,
				SampleCount: 160000, NumSpecies: 10000, EmbeddingDim: 1536, PreSigmoided: false,
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
				ModelType: ModelTypeBirdNetV30, SampleRate: 32000, Duration: 5.0,
				SampleCount: 160000, NumSpecies: 6522, EmbeddingDim: 1280, PreSigmoided: false,
			},
		},
		{
			name:    "dynamic input fallback v2.4",
			inputs:  []tensorInfo{{Name: "input", Dimensions: []int64{1, -1}}},
			outputs: []tensorInfo{{Name: "output", Dimensions: []int64{1, 6522}}},
			want: ModelConfig{
				ModelType: ModelTypeBirdNetV24, SampleRate: 48000, Duration: 3.0,
				SampleCount: 144000, NumSpecies: 6522, EmbeddingDim: 0, PreSigmoided: false,
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
				ModelType: ModelTypePerchV2, SampleRate: 32000, Duration: 5.0,
				SampleCount: 160000, NumSpecies: 10000, EmbeddingDim: 1536, PreSigmoided: false,
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := detectFromShapes(tt.inputs, tt.outputs)

			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Errorf("detectFromShapes() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}

			if err != nil {
				t.Fatalf("detectFromShapes() unexpected error: %v", err)
			}

			if got != tt.want {
				t.Errorf("detectFromShapes() =\n  %+v\nwant\n  %+v", got, tt.want)
			}
		})
	}
}

func TestDynamicBatchSupported(t *testing.T) {
	tests := []struct {
		name   string
		inputs []tensorInfo
		want   bool
	}{
		{
			name:   "fixed batch 1",
			inputs: []tensorInfo{{Name: "input", Dimensions: []int64{1, 144000}}},
			want:   false,
		},
		{
			name:   "dynamic batch -1",
			inputs: []tensorInfo{{Name: "input", Dimensions: []int64{-1, 144000}}},
			want:   true,
		},
		{
			name:   "batch greater than 1",
			inputs: []tensorInfo{{Name: "input", Dimensions: []int64{32, 144000}}},
			want:   true,
		},
		{
			name:   "empty inputs",
			inputs: []tensorInfo{},
			want:   false,
		},
		{
			name:   "empty dimensions",
			inputs: []tensorInfo{{Name: "input", Dimensions: []int64{}}},
			want:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := dynamicBatchSupported(tt.inputs)
			if got != tt.want {
				t.Errorf("dynamicBatchSupported() = %v, want %v", got, tt.want)
			}
		})
	}
}
