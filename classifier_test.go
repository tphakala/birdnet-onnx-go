package birdnet

import (
	"errors"
	"testing"
)

func TestNewClassifierMissingModelPath(t *testing.T) {
	_, err := NewClassifier(
		WithLabels([]string{"sp1", "sp2"}),
	)
	if !errors.Is(err, ErrModelPath) {
		t.Errorf("got %v, want ErrModelPath", err)
	}
}

func TestNewClassifierMissingLabels(t *testing.T) {
	_, err := NewClassifier(
		WithModelPath("nonexistent.onnx"),
	)
	if !errors.Is(err, ErrLabelsRequired) {
		t.Errorf("got %v, want ErrLabelsRequired", err)
	}
}

func TestNewClassifierEmptyModelPath(t *testing.T) {
	_, err := NewClassifier(
		WithModelPath(""),
		WithLabels([]string{"sp1"}),
	)
	if err == nil {
		t.Fatal("expected error for empty model path, got nil")
	}
}

func TestNewClassifierNoOptions(t *testing.T) {
	_, err := NewClassifier()
	if !errors.Is(err, ErrModelPath) {
		t.Errorf("got %v, want ErrModelPath", err)
	}
}

func TestNewClassifierModelPathOnlyNoLabels(t *testing.T) {
	// Model path is set but no labels source is provided.
	_, err := NewClassifier(
		WithModelPath("some_model.onnx"),
	)
	if !errors.Is(err, ErrLabelsRequired) {
		t.Errorf("got %v, want ErrLabelsRequired", err)
	}
}
