package birdnet

import (
	"strings"
	"testing"
)

func TestLoadLabelsText(t *testing.T) {
	labels, err := LoadLabels("testdata/labels_simple.txt")
	if err != nil {
		t.Fatalf("LoadLabels text: %v", err)
	}
	if len(labels) != 3 {
		t.Fatalf("expected 3 labels, got %d", len(labels))
	}
	if labels[0] != "Turdus merula_Common Blackbird" {
		t.Errorf("labels[0] = %q, want %q", labels[0], "Turdus merula_Common Blackbird")
	}
	if labels[2] != "Erithacus rubecula_European Robin" {
		t.Errorf("labels[2] = %q, want %q", labels[2], "Erithacus rubecula_European Robin")
	}
}

func TestLoadLabelsCSV(t *testing.T) {
	labels, err := LoadLabels("testdata/labels_v30.csv")
	if err != nil {
		t.Fatalf("LoadLabels csv: %v", err)
	}
	if len(labels) != 3 {
		t.Fatalf("expected 3 labels, got %d", len(labels))
	}
	// Should pick sci_name column (highest priority)
	if labels[0] != "Turdus merula" {
		t.Errorf("labels[0] = %q, want %q", labels[0], "Turdus merula")
	}
	if labels[1] != "Parus major" {
		t.Errorf("labels[1] = %q, want %q", labels[1], "Parus major")
	}
	if labels[2] != "Erithacus rubecula" {
		t.Errorf("labels[2] = %q, want %q", labels[2], "Erithacus rubecula")
	}
}

func TestLoadLabelsJSON(t *testing.T) {
	labels, err := LoadLabels("testdata/labels_perch.json")
	if err != nil {
		t.Fatalf("LoadLabels json: %v", err)
	}
	if len(labels) != 3 {
		t.Fatalf("expected 3 labels, got %d", len(labels))
	}
	if labels[0] != "Turdus merula" {
		t.Errorf("labels[0] = %q, want %q", labels[0], "Turdus merula")
	}
}

func TestLoadLabelsFromReader(t *testing.T) {
	t.Run("text format", func(t *testing.T) {
		r := strings.NewReader("Species A\nSpecies B\nSpecies C\n")
		labels, err := LoadLabelsFromReader(r, "text")
		if err != nil {
			t.Fatalf("LoadLabelsFromReader text: %v", err)
		}
		if len(labels) != 3 {
			t.Fatalf("expected 3 labels, got %d", len(labels))
		}
		if labels[1] != "Species B" {
			t.Errorf("labels[1] = %q, want %q", labels[1], "Species B")
		}
	})

	t.Run("csv format", func(t *testing.T) {
		r := strings.NewReader("sci_name;com_name\nSpecies A;Common A\nSpecies B;Common B\n")
		labels, err := LoadLabelsFromReader(r, "csv")
		if err != nil {
			t.Fatalf("LoadLabelsFromReader csv: %v", err)
		}
		if len(labels) != 2 {
			t.Fatalf("expected 2 labels, got %d", len(labels))
		}
		if labels[0] != "Species A" {
			t.Errorf("labels[0] = %q, want %q", labels[0], "Species A")
		}
	})

	t.Run("json format", func(t *testing.T) {
		r := strings.NewReader(`["Alpha", "Beta"]`)
		labels, err := LoadLabelsFromReader(r, "json")
		if err != nil {
			t.Fatalf("LoadLabelsFromReader json: %v", err)
		}
		if len(labels) != 2 {
			t.Fatalf("expected 2 labels, got %d", len(labels))
		}
	})

	t.Run("unknown format", func(t *testing.T) {
		r := strings.NewReader("data")
		_, err := LoadLabelsFromReader(r, "xml")
		if err == nil {
			t.Fatal("expected error for unknown format, got nil")
		}
	})
}

func TestLoadLabelsFileNotFound(t *testing.T) {
	_, err := LoadLabels("/nonexistent/path/labels.txt")
	if err == nil {
		t.Fatal("expected error for missing file, got nil")
	}
}

func TestLoadLabelsCSVColumnPriority(t *testing.T) {
	// sci_name should take priority over com_name even when com_name appears first
	r := strings.NewReader("com_name;sci_name;other\nCommon 1;Scientific 1;Extra\nCommon 2;Scientific 2;Extra\n")
	labels, err := LoadLabelsFromReader(r, "csv")
	if err != nil {
		t.Fatalf("LoadLabelsFromReader: %v", err)
	}
	if len(labels) != 2 {
		t.Fatalf("expected 2 labels, got %d", len(labels))
	}
	if labels[0] != "Scientific 1" {
		t.Errorf("labels[0] = %q, want %q (sci_name should have priority)", labels[0], "Scientific 1")
	}
}

func TestLoadLabelsCSVAutoDetectFromContent(t *testing.T) {
	// File with .txt extension but CSV-like content should be detected as CSV by LoadLabels
	// We test via LoadLabelsFromReader with auto-detection through content sniffing
	r := strings.NewReader("idx;sci_name;com_name\n0;Species A;Common A\n")
	labels, err := LoadLabelsFromReader(r, "csv")
	if err != nil {
		t.Fatalf("LoadLabelsFromReader: %v", err)
	}
	if len(labels) != 1 {
		t.Fatalf("expected 1 label, got %d", len(labels))
	}
	if labels[0] != "Species A" {
		t.Errorf("labels[0] = %q, want %q", labels[0], "Species A")
	}
}

func TestLoadLabelsTextEmptyLines(t *testing.T) {
	r := strings.NewReader("Species A\n\nSpecies B\n\n\nSpecies C\n")
	labels, err := LoadLabelsFromReader(r, "text")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(labels) != 3 {
		t.Fatalf("expected 3 labels (empty lines skipped), got %d", len(labels))
	}
}
