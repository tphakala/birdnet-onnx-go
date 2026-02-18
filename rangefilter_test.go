package birdnet

import (
	"errors"
	"testing"
)

func TestCalculateWeek(t *testing.T) {
	tests := []struct {
		name  string
		month int
		day   int
		want  float32
	}{
		{name: "January 1", month: 1, day: 1, want: 1},
		{name: "January 7", month: 1, day: 7, want: 1},
		{name: "January 8", month: 1, day: 8, want: 2},
		{name: "June 15", month: 6, day: 15, want: 23},
		{name: "December 31", month: 12, day: 31, want: 48},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CalculateWeek(tt.month, tt.day)
			if got != tt.want {
				t.Errorf("CalculateWeek(%d, %d) = %v, want %v", tt.month, tt.day, got, tt.want)
			}
		})
	}
}

func TestCalculateWeekBoundaries(t *testing.T) {
	tests := []struct {
		name  string
		month int
		day   int
		want  float32
	}{
		{name: "Minimum", month: 1, day: 1, want: 1},
		{name: "Maximum clamped", month: 12, day: 31, want: 48},
		{name: "June 1", month: 6, day: 1, want: 21},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CalculateWeek(tt.month, tt.day)
			if got != tt.want {
				t.Errorf("CalculateWeek(%d, %d) = %v, want %v", tt.month, tt.day, got, tt.want)
			}
		})
	}
}

func TestGetSpeciesScoresValidation(t *testing.T) {
	// Create a RangeFilter with nil session — we only test validation logic,
	// which runs before any ONNX call.
	rf := &RangeFilter{}

	tests := []struct {
		name    string
		lat     float32
		lon     float32
		week    float32
		wantErr error
	}{
		{name: "lat too low", lat: -91, lon: 0, week: 1, wantErr: ErrInvalidCoords},
		{name: "lat too high", lat: 91, lon: 0, week: 1, wantErr: ErrInvalidCoords},
		{name: "lon too low", lat: 0, lon: -181, week: 1, wantErr: ErrInvalidCoords},
		{name: "lon too high", lat: 0, lon: 181, week: 1, wantErr: ErrInvalidCoords},
		{name: "week too low", lat: 0, lon: 0, week: 0, wantErr: ErrInvalidWeek},
		{name: "week too high", lat: 0, lon: 0, week: 49, wantErr: ErrInvalidWeek},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := rf.GetSpeciesScores(tt.lat, tt.lon, tt.week)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("got error %v, want %v", err, tt.wantErr)
			}
		})
	}
}
