package birdnet

import (
	"testing"

	"github.com/stretchr/testify/assert"
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
			assert.Equal(t, tt.want, tt.mt.String())
		})
	}
}

func TestInputSizeError(t *testing.T) {
	err := &InputSizeError{Expected: 144000, Got: 100}

	assert.Equal(t, "birdnet: input segment size mismatch: expected 144000 samples, got 100", err.Error())
	assert.ErrorIs(t, err, ErrInputSize)
}
