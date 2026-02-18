//go:build integration

package birdnet_test

import (
	"context"
	"os"
	"testing"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/tphakala/birdnet-onnx-go"
)

func TestMain(m *testing.M) {
	// Try common library paths.
	paths := []string{
		"onnxruntime.so",
		"onnxruntime.dylib",
		"/usr/lib/onnxruntime.so",
		"/usr/local/lib/onnxruntime.so",
		"/usr/local/lib/onnxruntime.dylib",
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			ort.SetSharedLibraryPath(p)
			break
		}
	}
	// Also check ONNXRUNTIME_LIB env var.
	if envPath := os.Getenv("ONNXRUNTIME_LIB"); envPath != "" {
		ort.SetSharedLibraryPath(envPath)
	}

	if err := ort.InitializeEnvironment(); err != nil {
		panic("failed to initialize ONNX Runtime: " + err.Error())
	}
	code := m.Run()
	ort.DestroyEnvironment()
	os.Exit(code)
}

// skipIfMissing skips the test if the file at path does not exist.
func skipIfMissing(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("model file not found: %s", path)
	}
}

// modelPath returns the environment variable value if set, otherwise the default.
func modelPath(envKey, defaultPath string) string {
	if v := os.Getenv(envKey); v != "" {
		return v
	}
	return defaultPath
}

func TestBirdNetV24EndToEnd(t *testing.T) {
	model := modelPath("BIRDNET_V24_MODEL", "testdata/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx")
	labels := modelPath("BIRDNET_V24_LABELS", "testdata/BirdNET_GLOBAL_6K_V2.4_Labels_en.txt")
	skipIfMissing(t, model)

	classifier, err := birdnet.NewClassifier(
		birdnet.WithModelPath(model),
		birdnet.WithLabelsFromFile(labels),
		birdnet.WithTopK(5),
	)
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}
	defer classifier.Close()

	cfg := classifier.Config()
	if cfg.ModelType != birdnet.ModelTypeBirdNetV24 {
		t.Errorf("ModelType = %v, want %v", cfg.ModelType, birdnet.ModelTypeBirdNetV24)
	}
	if cfg.SampleRate != 48000 {
		t.Errorf("SampleRate = %d, want 48000", cfg.SampleRate)
	}
	if cfg.SampleCount != 144000 {
		t.Errorf("SampleCount = %d, want 144000", cfg.SampleCount)
	}

	// Predict with silence (all zeros).
	silence := make([]float32, cfg.SampleCount)
	result, err := classifier.Predict(context.Background(), silence)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	// Predictions may be empty if all below threshold; just verify non-nil result.
	if result == nil {
		t.Fatal("Predict returned nil result")
	}

	// v2.4 has no embeddings.
	if result.Embeddings != nil {
		t.Errorf("Embeddings should be nil for v2.4, got len %d", len(result.Embeddings))
	}

	// RawScores should have one entry per species.
	if len(result.RawScores) != cfg.NumSpecies {
		t.Errorf("len(RawScores) = %d, want %d", len(result.RawScores), cfg.NumSpecies)
	}
}

func TestBirdNetV30EndToEnd(t *testing.T) {
	model := modelPath("BIRDNET_V30_MODEL", "testdata/BirdNET_GLOBAL_V3.0_Model_FP32.onnx")
	labels := modelPath("BIRDNET_V30_LABELS", "testdata/BirdNET_GLOBAL_V3.0_Labels.csv")
	skipIfMissing(t, model)

	classifier, err := birdnet.NewClassifier(
		birdnet.WithModelPath(model),
		birdnet.WithLabelsFromFile(labels),
		birdnet.WithTopK(5),
	)
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}
	defer classifier.Close()

	cfg := classifier.Config()
	if cfg.ModelType != birdnet.ModelTypeBirdNetV30 {
		t.Errorf("ModelType = %v, want %v", cfg.ModelType, birdnet.ModelTypeBirdNetV30)
	}
	if cfg.SampleRate != 32000 {
		t.Errorf("SampleRate = %d, want 32000", cfg.SampleRate)
	}
	if cfg.SampleCount != 160000 {
		t.Errorf("SampleCount = %d, want 160000", cfg.SampleCount)
	}

	// Predict with silence.
	silence := make([]float32, cfg.SampleCount)
	result, err := classifier.Predict(context.Background(), silence)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	if result == nil {
		t.Fatal("Predict returned nil result")
	}

	// v3.0 should produce embeddings.
	if result.Embeddings == nil {
		t.Fatal("Embeddings should not be nil for v3.0")
	}
	if len(result.Embeddings) != cfg.EmbeddingDim {
		t.Errorf("len(Embeddings) = %d, want %d", len(result.Embeddings), cfg.EmbeddingDim)
	}
}

func TestPerchV2EndToEnd(t *testing.T) {
	model := modelPath("PERCH_V2_MODEL", "testdata/perch_v2.onnx")
	labels := modelPath("PERCH_V2_LABELS", "testdata/perch_v2_labels.csv")
	skipIfMissing(t, model)

	classifier, err := birdnet.NewClassifier(
		birdnet.WithModelPath(model),
		birdnet.WithLabelsFromFile(labels),
		birdnet.WithTopK(5),
	)
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}
	defer classifier.Close()

	cfg := classifier.Config()
	if cfg.ModelType != birdnet.ModelTypePerchV2 {
		t.Errorf("ModelType = %v, want %v", cfg.ModelType, birdnet.ModelTypePerchV2)
	}
	if cfg.SampleRate != 32000 {
		t.Errorf("SampleRate = %d, want 32000", cfg.SampleRate)
	}
	if cfg.SampleCount != 160000 {
		t.Errorf("SampleCount = %d, want 160000", cfg.SampleCount)
	}

	// Predict with silence.
	silence := make([]float32, cfg.SampleCount)
	result, err := classifier.Predict(context.Background(), silence)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	if result == nil {
		t.Fatal("Predict returned nil result")
	}

	// Perch should produce embeddings.
	if result.Embeddings == nil {
		t.Fatal("Embeddings should not be nil for Perch v2")
	}
}

func TestBSGFinlandEndToEnd(t *testing.T) {
	model := modelPath("BSG_MODEL", "testdata/bsg_v4.4.onnx")
	labels := modelPath("BSG_LABELS", "testdata/bsg_labels.txt")
	skipIfMissing(t, model)

	classifier, err := birdnet.NewClassifier(
		birdnet.WithModelPath(model),
		birdnet.WithLabelsFromFile(labels),
		birdnet.WithModelType(birdnet.ModelTypeBSGFinland),
		birdnet.WithTopK(5),
	)
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}
	defer classifier.Close()

	cfg := classifier.Config()
	if cfg.ModelType != birdnet.ModelTypeBSGFinland {
		t.Errorf("ModelType = %v, want %v", cfg.ModelType, birdnet.ModelTypeBSGFinland)
	}
	if !cfg.PreSigmoided {
		t.Error("PreSigmoided should be true for BSG Finland")
	}

	// Predict with silence.
	silence := make([]float32, cfg.SampleCount)
	result, err := classifier.Predict(context.Background(), silence)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	if result == nil {
		t.Fatal("Predict returned nil result")
	}

	// All raw scores should be in [0, 1] (pre-sigmoided).
	for i, score := range result.RawScores {
		if score < 0 || score > 1 {
			t.Errorf("RawScores[%d] = %f, want value in [0, 1]", i, score)
		}
	}
}

func TestBatchPrediction(t *testing.T) {
	model := modelPath("BIRDNET_V24_MODEL", "testdata/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx")
	labels := modelPath("BIRDNET_V24_LABELS", "testdata/BirdNET_GLOBAL_6K_V2.4_Labels_en.txt")
	skipIfMissing(t, model)

	classifier, err := birdnet.NewClassifier(
		birdnet.WithModelPath(model),
		birdnet.WithLabelsFromFile(labels),
		birdnet.WithTopK(5),
	)
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}
	defer classifier.Close()

	cfg := classifier.Config()

	// Create 4 silence segments.
	segments := make([][]float32, 4)
	for i := range segments {
		segments[i] = make([]float32, cfg.SampleCount)
	}

	results, err := classifier.PredictBatch(context.Background(), segments)
	if err != nil {
		t.Fatalf("PredictBatch: %v", err)
	}

	if len(results) != 4 {
		t.Errorf("len(results) = %d, want 4", len(results))
	}
}

func TestRangeFilterEndToEnd(t *testing.T) {
	metaModel := modelPath("BIRDNET_META_MODEL", "testdata/BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.onnx")
	labelsFile := modelPath("BIRDNET_V24_LABELS", "testdata/BirdNET_GLOBAL_6K_V2.4_Labels_en.txt")
	skipIfMissing(t, metaModel)

	labels, err := birdnet.LoadLabels(labelsFile)
	if err != nil {
		t.Fatalf("LoadLabels: %v", err)
	}

	rf, err := birdnet.NewRangeFilter(metaModel, labels)
	if err != nil {
		t.Fatalf("NewRangeFilter: %v", err)
	}
	defer rf.Close()

	// Helsinki: lat 60.17, lon 24.94, week 22.
	scores, err := rf.GetSpeciesScores(60.17, 24.94, 22)
	if err != nil {
		t.Fatalf("GetSpeciesScores: %v", err)
	}

	if len(scores) != len(labels) {
		t.Errorf("len(scores) = %d, want %d", len(scores), len(labels))
	}

	for i, s := range scores {
		if s < 0 || s > 1 {
			t.Errorf("scores[%d] = %f, want value in [0, 1]", i, s)
		}
	}
}

func TestContextCancellation(t *testing.T) {
	model := modelPath("BIRDNET_V24_MODEL", "testdata/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx")
	labels := modelPath("BIRDNET_V24_LABELS", "testdata/BirdNET_GLOBAL_6K_V2.4_Labels_en.txt")
	skipIfMissing(t, model)

	classifier, err := birdnet.NewClassifier(
		birdnet.WithModelPath(model),
		birdnet.WithLabelsFromFile(labels),
	)
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}
	defer classifier.Close()

	cfg := classifier.Config()
	silence := make([]float32, cfg.SampleCount)

	// Create an already-cancelled context.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = classifier.Predict(ctx, silence)
	if err == nil {
		t.Fatal("Predict with cancelled context should return an error")
	}
}
