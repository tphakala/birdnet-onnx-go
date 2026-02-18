package birdnet

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// LoadLabels reads a label file and returns labels as a string slice.
// The format is auto-detected from the file extension and content:
//   - .json -> JSON array of strings
//   - .csv  -> CSV with smart column detection
//   - otherwise -> plain text (one label per line), with a content-sniff
//     fallback to CSV if the first line looks like a delimited header.
func LoadLabels(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("birdnet: open label file: %w", err)
	}
	defer f.Close()

	format := detectFormat(path, f)

	// After sniffing, reset to the beginning so the loader reads full content.
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, fmt.Errorf("birdnet: seek label file: %w", err)
	}

	return LoadLabelsFromReader(f, format)
}

// LoadLabelsFromReader parses labels from an io.Reader using the given format.
// Supported formats: "text", "csv", "json".
// This is useful when labels are embedded with go:embed.
func LoadLabelsFromReader(r io.Reader, format string) ([]string, error) {
	switch strings.ToLower(format) {
	case "text":
		return loadText(r)
	case "csv":
		return loadCSV(r)
	case "json":
		return loadJSON(r)
	default:
		return nil, fmt.Errorf("birdnet: unsupported label format %q", format)
	}
}

// detectFormat determines the label format from the file extension and,
// if ambiguous, by peeking at the first line of the file.
func detectFormat(path string, r io.ReadSeeker) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".json":
		return "json"
	case ".csv":
		return "csv"
	}

	// Sniff the first line to decide between CSV and plain text.
	scanner := bufio.NewScanner(r)
	if scanner.Scan() {
		line := scanner.Text()
		if looksLikeCSV(line) {
			return "csv"
		}
	}
	return "text"
}

// looksLikeCSV returns true if the line contains delimiters that split it into
// multiple fields, suggesting CSV content.
func looksLikeCSV(line string) bool {
	semicolons := strings.Count(line, ";")
	commas := strings.Count(line, ",")
	return semicolons >= 1 || commas >= 1
}

// loadText parses one-label-per-line format, skipping empty lines.
func loadText(r io.Reader) ([]string, error) {
	var labels []string
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			labels = append(labels, line)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("birdnet: read text labels: %w", err)
	}
	return labels, nil
}

// loadCSV parses CSV/semicolon-separated label files with smart column detection.
// It auto-detects the delimiter and, if a header row is present, selects the
// best label column using priority: sci_name > com_name > species > name > label.
func loadCSV(r io.Reader) ([]string, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("birdnet: read csv labels: %w", err)
	}
	content := string(data)

	// Strip UTF-8 BOM if present.
	content = strings.TrimPrefix(content, "\xef\xbb\xbf")

	delimiter := detectDelimiter(content)

	reader := csv.NewReader(strings.NewReader(content))
	reader.Comma = delimiter
	reader.LazyQuotes = true
	reader.FieldsPerRecord = -1 // allow variable field counts

	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("birdnet: parse csv labels: %w", err)
	}
	if len(records) == 0 {
		return nil, nil
	}

	colIdx := 0
	startRow := 0

	// Check if the first row is a header.
	if isHeaderRow(records[0]) {
		colIdx = findLabelColumn(records[0])
		startRow = 1
	}

	var labels []string
	for _, row := range records[startRow:] {
		if colIdx >= len(row) {
			continue
		}
		label := strings.TrimSpace(row[colIdx])
		if label != "" {
			labels = append(labels, label)
		}
	}
	return labels, nil
}

// loadJSON parses a JSON array of strings.
func loadJSON(r io.Reader) ([]string, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("birdnet: read json labels: %w", err)
	}

	var labels []string
	if err := json.Unmarshal(data, &labels); err != nil {
		return nil, fmt.Errorf("birdnet: parse json labels: %w", err)
	}
	return labels, nil
}

// detectDelimiter checks the first line to decide between semicolon and comma.
func detectDelimiter(content string) rune {
	line, _, _ := strings.Cut(content, "\n")
	if strings.Count(line, ";") > strings.Count(line, ",") {
		return ';'
	}
	return ','
}

// headerKeywords are values that indicate a CSV row is a header, not data.
var headerKeywords = map[string]bool{
	"idx":             true,
	"id":              true,
	"label":           true,
	"name":            true,
	"species":         true,
	"class":           true,
	"sci_name":        true,
	"com_name":        true,
	"scientific_name": true,
	"common_name":     true,
	"family":          true,
	"order":           true,
}

// isHeaderRow returns true if the first field of the row looks like a known
// CSV header keyword.
func isHeaderRow(row []string) bool {
	if len(row) == 0 {
		return false
	}
	first := strings.TrimSpace(strings.ToLower(row[0]))
	return headerKeywords[first]
}

// columnPriority lists header names in descending priority for label selection.
var columnPriority = []string{
	"sci_name",
	"com_name",
	"species",
	"name",
	"label",
}

// findLabelColumn returns the index of the best label column in a header row.
// It checks columnPriority in order and falls back to column 0.
func findLabelColumn(header []string) int {
	norm := make([]string, len(header))
	for i, h := range header {
		norm[i] = strings.TrimSpace(strings.ToLower(h))
	}

	for _, want := range columnPriority {
		for i, h := range norm {
			if h == want {
				return i
			}
		}
	}
	return 0
}
