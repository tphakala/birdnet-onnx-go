package birdnet

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// Range filter model constants.
const (
	rangeFilterInputDims = 3  // lat, lon, week
	maxWeeksPerYear      = 48 // number of weeks in the range filter calendar
)

// CalculateWeek converts a month and day to a week number in the range [1, 48].
// The formula is: week = (month-1)*4 + (day-1)/7 + 1, clamped to [1, 48].
func CalculateWeek(month, day int) float32 {
	week := max((month-1)*4+(day-1)/7+1, 1)
	week = min(week, maxWeeksPerYear)
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
	numSpecies, err := extractOutputSize(ortOutputs[0].Dimensions)
	if err != nil {
		return nil, err
	}

	// Validate label count against model output.
	if int64(len(labels)) != numSpecies {
		return nil, fmt.Errorf("%w: got %d labels, range filter model expects %d",
			ErrLabelCount, len(labels), numSpecies)
	}

	// Create pre-allocated input tensor: shape [1, 3].
	inputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, rangeFilterInputDims))
	if err != nil {
		return nil, fmt.Errorf("birdnet: creating range filter input tensor: %w", err)
	}

	// Create pre-allocated output tensor: shape [1, numSpecies].
	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, numSpecies))
	if err != nil {
		_ = inputTensor.Destroy()
		return nil, fmt.Errorf("birdnet: creating range filter output tensor: %w", err)
	}

	inputs := []ort.Value{inputTensor}
	outputs := []ort.Value{outputTensor}

	session, err := ort.NewAdvancedSession(
		modelPath, inputNames, outputNames, inputs, outputs, nil,
	)
	if err != nil {
		_ = inputTensor.Destroy()
		_ = outputTensor.Destroy()
		return nil, fmt.Errorf("birdnet: creating range filter ONNX session: %w", err)
	}

	return &RangeFilter{
		session:      session,
		labels:       labels,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
	}, nil
}

// minOutputDims is the minimum number of dimensions needed to extract the species count
// from the second axis of the output tensor (e.g. shape [1, numSpecies]).
const minOutputDims = 2

// extractOutputSize reads the species count from the output dimensions.
func extractOutputSize(dims ort.Shape) (int64, error) {
	var size int64
	switch {
	case len(dims) >= minOutputDims:
		size = dims[1]
	case len(dims) == 1:
		size = dims[0]
	default:
		return 0, fmt.Errorf("birdnet: unexpected output shape in range filter model")
	}
	if size <= 0 {
		return 0, fmt.Errorf("birdnet: invalid species count %d in range filter output shape %v", size, dims)
	}
	return size, nil
}

// GetSpeciesScores runs the range filter model and returns a score for each
// species indicating presence likelihood at the given location and week.
//
// Latitude must be in [-90, 90], longitude in [-180, 180], and week in [1, 48].
func (rf *RangeFilter) GetSpeciesScores(lat, lon, week float32) ([]float32, error) {
	if lat < -90 || lat > 90 || lon < -180 || lon > 180 {
		return nil, fmt.Errorf("%w: lat=%v, lon=%v", ErrInvalidCoords, lat, lon)
	}
	if week < 1 || week > maxWeeksPerYear {
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
			_ = rf.session.Destroy()
		}
		if rf.outputTensor != nil {
			_ = rf.outputTensor.Destroy()
		}
		if rf.inputTensor != nil {
			_ = rf.inputTensor.Destroy()
		}
	})
	return nil
}
