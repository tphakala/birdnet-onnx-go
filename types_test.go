package birdnet

import (
	"errors"
	"testing"
)

func TestModelTypeString(t *testing.T) {
	tests := []struct {
		mt   ModelType
		want string
	}{
		{ModelTypeBirdNetV24, "BirdNET v2.4"},
		{ModelTypeBirdNetV30, "BirdNET v3.0"},
		{ModelTypePerchV2, "Perch v2"},
		{ModelTypeBSGFinland, "BSG Finland"},
		{ModelType(99), "ModelType(99)"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			got := tt.mt.String()
			if got != tt.want {
				t.Errorf("ModelType(%d).String() = %q, want %q", int(tt.mt), got, tt.want)
			}
		})
	}
}

func TestInputSizeError(t *testing.T) {
	err := &InputSizeError{Expected: 144000, Got: 100}

	// Verify error message contains expected and got values.
	msg := err.Error()
	if msg != "birdnet: input segment size mismatch: expected 144000 samples, got 100" {
		t.Errorf("unexpected error message: %s", msg)
	}

	// Verify Unwrap returns ErrInputSize.
	if !errors.Is(err, ErrInputSize) {
		t.Error("InputSizeError should unwrap to ErrInputSize")
	}
}
