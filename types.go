package birdnet

import "fmt"

// Sample count constants define the expected number of float32 samples per audio segment
// for each supported model type.
const (
	SampleCountV24   = 144_000 // 48kHz * 3.0s
	SampleCountV30   = 160_000 // 32kHz * 5.0s
	SampleCountPerch = 160_000 // 32kHz * 5.0s
	SampleCountBSG   = 144_000 // 48kHz * 3.0s
)

// ModelType identifies which ONNX model variant is in use.
type ModelType int

const (
	ModelTypeBirdNetV24 ModelType = iota
	ModelTypeBirdNetV30
	ModelTypePerchV2
	ModelTypeBSGFinland
)

// String returns a human-readable name for the model type.
func (m ModelType) String() string {
	switch m {
	case ModelTypeBirdNetV24:
		return "BirdNET v2.4"
	case ModelTypeBirdNetV30:
		return "BirdNET v3.0"
	case ModelTypePerchV2:
		return "Perch v2"
	case ModelTypeBSGFinland:
		return "BSG Finland"
	default:
		return fmt.Sprintf("ModelType(%d)", int(m))
	}
}

// ModelConfig holds the detected configuration for a loaded ONNX model.
type ModelConfig struct {
	ModelType    ModelType
	SampleRate   int     // 48000 or 32000
	Duration     float64 // 3.0 or 5.0 seconds
	SampleCount  int     // 144000 or 160000
	NumSpecies   int     // from model output shape
	EmbeddingDim int     // 0 for v2.4/BSG, 1280 for v3.0, variable for Perch
	PreSigmoided bool    // true for BSG only (scores already in [0,1])
}

// Prediction represents a single species prediction with its confidence score.
type Prediction struct {
	Species    string  // Label text (e.g. "Turdus merula_Common Blackbird")
	Confidence float32 // 0.0-1.0 after sigmoid/identity
	Index      int     // Position in the label array
}

// PredictionResult contains the full output from a model inference pass.
type PredictionResult struct {
	ModelType   ModelType
	Predictions []Prediction // Top-K, sorted by confidence descending
	Embeddings  []float32    // nil for v2.4 and BSG
	RawScores   []float32    // All species scores after activation
}
