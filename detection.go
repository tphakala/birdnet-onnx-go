package birdnet

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// tensorInfo mirrors the shape fields from ort.InputOutputInfo needed for detection.
type tensorInfo struct {
	Name       string
	Dimensions []int64
}

// DetectModelType inspects the ONNX model at modelPath and returns the detected ModelConfig.
// It calls ort.GetInputOutputInfo to read tensor shapes, then delegates to detectFromShapes.
func DetectModelType(modelPath string) (ModelConfig, error) {
	ortInputs, ortOutputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return ModelConfig{}, fmt.Errorf("birdnet: reading ONNX model info: %w", err)
	}

	inputs := make([]tensorInfo, len(ortInputs))
	for i, info := range ortInputs {
		inputs[i] = tensorInfo{
			Name:       info.Name,
			Dimensions: []int64(info.Dimensions),
		}
	}

	outputs := make([]tensorInfo, len(ortOutputs))
	for i, info := range ortOutputs {
		outputs[i] = tensorInfo{
			Name:       info.Name,
			Dimensions: []int64(info.Dimensions),
		}
	}

	return detectFromShapes(inputs, outputs)
}

// detectFromShapes determines the model type from input/output tensor shapes.
// This is a pure function with no ONNX dependency, making it easily testable.
func detectFromShapes(inputs, outputs []tensorInfo) (ModelConfig, error) {
	if len(inputs) == 0 || len(outputs) == 0 {
		return ModelConfig{}, ErrModelDetection
	}

	inputDims := inputs[0].Dimensions
	if len(inputDims) == 0 {
		return ModelConfig{}, ErrModelDetection
	}

	sampleCount := inputDims[len(inputDims)-1]
	outputCount := len(outputs)

	// Try exact match on (sampleCount, outputCount) first.
	cfg, ok := matchModel(sampleCount, outputCount, outputs)
	if ok {
		return cfg, nil
	}

	// If the input sample dimension is dynamic (<=0), fall back to output count alone.
	if sampleCount <= 0 {
		cfg, ok = matchByOutputCount(outputCount, outputs)
		if ok {
			return cfg, nil
		}
	}

	return ModelConfig{}, ErrModelDetection
}

// matchModel attempts to match the model type from a known (sampleCount, outputCount) pair.
func matchModel(sampleCount int64, outputCount int, outputs []tensorInfo) (ModelConfig, bool) {
	switch {
	case sampleCount == SampleCountV24 && outputCount == 1:
		return ModelConfig{
			ModelType:    ModelTypeBirdNetV24,
			SampleRate:   48000,
			Duration:     3.0,
			SampleCount:  SampleCountV24,
			NumSpecies:   lastDim(outputs[0]),
			EmbeddingDim: 0,
			PreSigmoided: false,
		}, true

	case sampleCount == SampleCountV30 && outputCount == 2:
		return ModelConfig{
			ModelType:    ModelTypeBirdNetV30,
			SampleRate:   32000,
			Duration:     5.0,
			SampleCount:  SampleCountV30,
			NumSpecies:   lastDim(outputs[1]),
			EmbeddingDim: lastDim(outputs[0]),
			PreSigmoided: false,
		}, true

	case sampleCount == SampleCountPerch && outputCount == 4:
		return ModelConfig{
			ModelType:    ModelTypePerchV2,
			SampleRate:   32000,
			Duration:     5.0,
			SampleCount:  SampleCountPerch,
			NumSpecies:   lastDim(outputs[1]),
			EmbeddingDim: lastDim(outputs[0]),
			PreSigmoided: false,
		}, true

	default:
		return ModelConfig{}, false
	}
}

// matchByOutputCount falls back to output count alone when input dimensions are dynamic.
func matchByOutputCount(outputCount int, outputs []tensorInfo) (ModelConfig, bool) {
	switch outputCount {
	case 1:
		return ModelConfig{
			ModelType:    ModelTypeBirdNetV24,
			SampleRate:   48000,
			Duration:     3.0,
			SampleCount:  SampleCountV24,
			NumSpecies:   lastDim(outputs[0]),
			EmbeddingDim: 0,
			PreSigmoided: false,
		}, true

	case 2:
		return ModelConfig{
			ModelType:    ModelTypeBirdNetV30,
			SampleRate:   32000,
			Duration:     5.0,
			SampleCount:  SampleCountV30,
			NumSpecies:   lastDim(outputs[1]),
			EmbeddingDim: lastDim(outputs[0]),
			PreSigmoided: false,
		}, true

	case 4:
		return ModelConfig{
			ModelType:    ModelTypePerchV2,
			SampleRate:   32000,
			Duration:     5.0,
			SampleCount:  SampleCountPerch,
			NumSpecies:   lastDim(outputs[1]),
			EmbeddingDim: lastDim(outputs[0]),
			PreSigmoided: false,
		}, true

	default:
		return ModelConfig{}, false
	}
}

// dynamicBatchSupported checks if the model's input batch dimension is dynamic (<=0 or >1).
func dynamicBatchSupported(inputs []tensorInfo) bool {
	if len(inputs) == 0 {
		return false
	}
	dims := inputs[0].Dimensions
	if len(dims) == 0 {
		return false
	}
	return dims[0] <= 0 || dims[0] > 1
}

// lastDim returns the last dimension of a tensorInfo as an int, or 0 if the dimensions are empty.
func lastDim(t tensorInfo) int {
	if len(t.Dimensions) == 0 {
		return 0
	}
	return int(t.Dimensions[len(t.Dimensions)-1])
}
