package birdnet

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadLabelsText(t *testing.T) {
	labels, err := LoadLabels("testdata/labels_simple.txt")
	require.NoError(t, err)
	require.Len(t, labels, 3)
	assert.Equal(t, "Turdus merula_Common Blackbird", labels[0])
	assert.Equal(t, "Erithacus rubecula_European Robin", labels[2])
}

func TestLoadLabelsCSV(t *testing.T) {
	labels, err := LoadLabels("testdata/labels_v30.csv")
	require.NoError(t, err)
	require.Len(t, labels, 3)
	// Should pick sci_name column (highest priority)
	assert.Equal(t, "Turdus merula", labels[0])
	assert.Equal(t, "Parus major", labels[1])
	assert.Equal(t, "Erithacus rubecula", labels[2])
}

func TestLoadLabelsJSON(t *testing.T) {
	labels, err := LoadLabels("testdata/labels_perch.json")
	require.NoError(t, err)
	require.Len(t, labels, 3)
	assert.Equal(t, "Turdus merula", labels[0])
}

func TestLoadLabelsFromReader(t *testing.T) {
	t.Run("text format", func(t *testing.T) {
		r := strings.NewReader("Species A\nSpecies B\nSpecies C\n")
		labels, err := LoadLabelsFromReader(r, "text")
		require.NoError(t, err)
		require.Len(t, labels, 3)
		assert.Equal(t, "Species B", labels[1])
	})

	t.Run("csv format", func(t *testing.T) {
		r := strings.NewReader("sci_name;com_name\nSpecies A;Common A\nSpecies B;Common B\n")
		labels, err := LoadLabelsFromReader(r, "csv")
		require.NoError(t, err)
		require.Len(t, labels, 2)
		assert.Equal(t, "Species A", labels[0])
	})

	t.Run("json format", func(t *testing.T) {
		r := strings.NewReader(`["Alpha", "Beta"]`)
		labels, err := LoadLabelsFromReader(r, "json")
		require.NoError(t, err)
		require.Len(t, labels, 2)
	})

	t.Run("unknown format", func(t *testing.T) {
		r := strings.NewReader("data")
		_, err := LoadLabelsFromReader(r, "xml")
		assert.Error(t, err)
	})
}

func TestLoadLabelsFileNotFound(t *testing.T) {
	_, err := LoadLabels("/nonexistent/path/labels.txt")
	assert.Error(t, err)
}

func TestLoadLabelsCSVColumnPriority(t *testing.T) {
	// sci_name should take priority over com_name even when com_name appears first
	r := strings.NewReader("com_name;sci_name;other\nCommon 1;Scientific 1;Extra\nCommon 2;Scientific 2;Extra\n")
	labels, err := LoadLabelsFromReader(r, "csv")
	require.NoError(t, err)
	require.Len(t, labels, 2)
	assert.Equal(t, "Scientific 1", labels[0], "sci_name should have priority")
}

func TestLoadLabelsCSVAutoDetectFromContent(t *testing.T) {
	r := strings.NewReader("idx;sci_name;com_name\n0;Species A;Common A\n")
	labels, err := LoadLabelsFromReader(r, "csv")
	require.NoError(t, err)
	require.Len(t, labels, 1)
	assert.Equal(t, "Species A", labels[0])
}

func TestLooksLikeCSV(t *testing.T) {
	tests := []struct {
		line string
		want bool
	}{
		{"Turdus merula_Common Blackbird", false},       // plain text label
		{"Parus major, Great Tit", false},               // single comma in species name
		{"idx;sci_name;com_name", true},                 // CSV header with semicolons
		{"idx,sci_name,com_name", true},                 // CSV header with commas
		{"0;Turdus merula;Common Blackbird", true},      // CSV data row
		{"species_a, something; else, more", true},      // mixed delimiters
		{"", false},                                     // empty line
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, looksLikeCSV(tt.line), "looksLikeCSV(%q)", tt.line)
	}
}

func TestLoadLabelsTextEmptyLines(t *testing.T) {
	r := strings.NewReader("Species A\n\nSpecies B\n\n\nSpecies C\n")
	labels, err := LoadLabelsFromReader(r, "text")
	require.NoError(t, err)
	require.Len(t, labels, 3, "empty lines should be skipped")
}
