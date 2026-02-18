package birdnet

import (
	"errors"
	"fmt"
)

// Sentinel errors returned by the classifier and related functions.
var (
	ErrInputSize      = errors.New("birdnet: input segment size mismatch")
	ErrBatchInputSize = errors.New("birdnet: batch segment size mismatch")
	ErrModelDetection = errors.New("birdnet: unable to detect model type from ONNX shapes")
	ErrLabelCount     = errors.New("birdnet: label count does not match model output dimension")
	ErrModelPath      = errors.New("birdnet: model path is required")
	ErrLabelsRequired = errors.New("birdnet: labels are required")
	ErrInvalidCoords  = errors.New("birdnet: invalid coordinates")
	ErrInvalidWeek    = errors.New("birdnet: week must be between 1 and 48")
)

// InputSizeError provides detailed information about a sample count mismatch.
type InputSizeError struct {
	Expected int
	Got      int
}

// Error returns a descriptive message including expected and actual sizes.
func (e *InputSizeError) Error() string {
	return fmt.Sprintf("birdnet: input segment size mismatch: expected %d samples, got %d", e.Expected, e.Got)
}

// Unwrap returns ErrInputSize so callers can use errors.Is for matching.
func (e *InputSizeError) Unwrap() error {
	return ErrInputSize
}
