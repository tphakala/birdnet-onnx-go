package birdnet

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// CalculateWeek converts a month and day to a week number in the range [1, 48].
// The formula is: week = (month-1)*4 + (day-1)/7 + 1, clamped to [1, 48].
func CalculateWeek(month, day int) float32 {
	week := max((month-1)*4+(day-1)/7+1, 1)
	week = min(week, 48)
	return float32(week)
}

// RangeFilter wraps an ONNX Runtime session for species range filtering.
// Given a latitude, longitude, and week of the year, it produces a score
// per species indicating how likely that species is to be present.
// It is safe for concurrent use.
type RangeFilter struct {
	session      *ort.AdvancedSession
	labels       []string
	mu           sync.Mutex
	closeOnce    sync.Once
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
}

// NewRangeFilter creates a new RangeFilter from the given ONNX model path
// and species labels. The model must accept input shape [1, 3] (lat, lon, week)
// and produce output shape [1, numSpecies].
func NewRangeFilter(modelPath string, labels []string) (*RangeFilter, error) {
	ortInputs, ortOutputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, fmt.Errorf("birdnet: reading range filter model info: %w", err)
	}

	inputNames := make([]string, len(ortInputs))
	for i, info := range ortInputs {
		inputNames[i] = info.Name
	}
	outputNames := make([]string, len(ortOutputs))
	for i, info := range ortOutputs {
		outputNames[i] = info.Name
	}

	// Determine output size from model metadata.
	if len(ortOutputs) == 0 {
		return nil, fmt.Errorf("birdnet: range filter model has no outputs")
	}
	outputDims := ortOutputs[0].Dimensions
	var numSpecies int64
	if len(outputDims) >= 2 {
		numSpecies = outputDims[1]
	} else if len(outputDims) == 1 {
		numSpecies = outputDims[0]
	} else {
		return nil, fmt.Errorf("birdnet: unexpected output shape in range filter model")
	}

	// Create pre-allocated input tensor: shape [1, 3].
	inputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 3))
	if err != nil {
		return nil, fmt.Errorf("birdnet: creating range filter input tensor: %w", err)
	}

	// Create pre-allocated output tensor: shape [1, numSpecies].
	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, numSpecies))
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("birdnet: creating range filter output tensor: %w", err)
	}

	inputs := []ort.Value{inputTensor}
	outputs := []ort.Value{outputTensor}

	session, err := ort.NewAdvancedSession(
		modelPath, inputNames, outputNames, inputs, outputs, nil,
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("birdnet: creating range filter ONNX session: %w", err)
	}

	return &RangeFilter{
		session:      session,
		labels:       labels,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
	}, nil
}

// GetSpeciesScores runs the range filter model and returns a score for each
// species indicating presence likelihood at the given location and week.
//
// Latitude must be in [-90, 90], longitude in [-180, 180], and week in [1, 48].
func (rf *RangeFilter) GetSpeciesScores(lat, lon, week float32) ([]float32, error) {
	if lat < -90 || lat > 90 || lon < -180 || lon > 180 {
		return nil, fmt.Errorf("%w: lat=%v, lon=%v", ErrInvalidCoords, lat, lon)
	}
	if week < 1 || week > 48 {
		return nil, fmt.Errorf("%w: week=%v", ErrInvalidWeek, week)
	}

	rf.mu.Lock()
	defer rf.mu.Unlock()

	// Set input data: [lat, lon, week].
	data := rf.inputTensor.GetData()
	data[0] = lat
	data[1] = lon
	data[2] = week

	// Run inference.
	if err := rf.session.Run(); err != nil {
		return nil, fmt.Errorf("birdnet: running range filter inference: %w", err)
	}

	// Copy output data (don't retain tensor buffer reference).
	outputData := rf.outputTensor.GetData()
	scores := make([]float32, len(outputData))
	copy(scores, outputData)

	return scores, nil
}

// Close releases all ONNX Runtime resources held by the RangeFilter.
// It is safe to call Close multiple times.
func (rf *RangeFilter) Close() error {
	rf.closeOnce.Do(func() {
		if rf.session != nil {
			rf.session.Destroy()
		}
		if rf.outputTensor != nil {
			rf.outputTensor.Destroy()
		}
		if rf.inputTensor != nil {
			rf.inputTensor.Destroy()
		}
	})
	return nil
}
