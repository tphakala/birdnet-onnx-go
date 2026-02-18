package birdnet

import (
	"testing"

	"github.com/stretchr/testify/assert"
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
