# birdnet-onnx-go Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Go library for bird species detection using ONNX Runtime, supporting BirdNET v2.4/v3.0, Google Perch v2, and BSG Finland v4.4 models.

**Architecture:** Single `birdnet` package with functional options pattern. Unified `Classifier` struct auto-detects model type from ONNX tensor shapes. Uses `onnxruntime_go` for inference with pre-allocated tensor buffers and mutex-based thread safety.

**Tech Stack:** Go 1.26, github.com/yalue/onnxruntime_go v1.26.0

**Design doc:** `docs/plans/2026-02-18-birdnet-onnx-go-design.md`

**Reference implementation:** `../rust-birdnet-onnx/` (Rust)

---

### Task 1: Project Initialization

**Files:**
- Create: `go.mod`
- Create: `doc.go`

**Step 1: Initialize Go module**

Run:
```bash
cd /Users/e909385/src/birdnet-onnx-go
git init
go mod init github.com/tphakala/birdnet-onnx-go
```
Expected: `go.mod` created with `go 1.26`

**Step 2: Add onnxruntime_go dependency**

Run:
```bash
go get github.com/yalue/onnxruntime_go@latest
```
Expected: Dependency added to `go.mod` and `go.sum`

**Step 3: Create package doc**

Create `doc.go`:
```go
// Package birdnet provides bird species detection using ONNX Runtime models.
//
// It supports BirdNET v2.4, BirdNET v3.0, Google Perch v2, and BSG Finland v4.4
// models with automatic model type detection from ONNX tensor shapes.
//
// The caller is responsible for ONNX Runtime lifecycle management:
//
//	ort.SetSharedLibraryPath("/path/to/onnxruntime.so")
//	ort.InitializeEnvironment()
//	defer ort.DestroyEnvironment()
//
// Then create a classifier:
//
//	c, err := birdnet.NewClassifier(
//	    birdnet.WithModelPath("model.onnx"),
//	    birdnet.WithLabelsFromFile("labels.txt"),
//	)
//	defer c.Close()
//
//	result, err := c.Predict(ctx, audioSegment)
package birdnet
```

**Step 4: Commit**

```bash
git add go.mod go.sum doc.go
git commit -m "feat: initialize birdnet-onnx-go module"
```

---

### Task 2: Core Types and Errors

**Files:**
- Create: `types.go`
- Create: `errors.go`
- Create: `types_test.go`

**Step 1: Write types**

Create `types.go` with all core types from the design doc:
- `ModelType` enum with `String()` method
- `ModelConfig` struct
- `Prediction` struct
- `PredictionResult` struct

Reference: `../rust-birdnet-onnx/src/types.rs` for exact field names and model constants.

Model constants to define:
```go
const (
    SampleCountV24  = 144_000 // 48kHz * 3.0s
    SampleCountV30  = 160_000 // 32kHz * 5.0s
    SampleCountPerch = 160_000 // 32kHz * 5.0s
    SampleCountBSG  = 144_000 // 48kHz * 3.0s
)
```

**Step 2: Write errors**

Create `errors.go` with all sentinel errors from the design doc plus the `InputSizeError` type implementing `error` and `Unwrap()`.

**Step 3: Write tests for ModelType.String()**

Create `types_test.go`:
```go
func TestModelTypeString(t *testing.T) {
    tests := []struct {
        mt   ModelType
        want string
    }{
        {ModelTypeBirdNetV24, "BirdNET v2.4"},
        {ModelTypeBirdNetV30, "BirdNET v3.0"},
        {ModelTypePerchV2, "Perch v2"},
        {ModelTypeBSGFinland, "BSG Finland"},
    }
    for _, tt := range tests {
        if got := tt.mt.String(); got != tt.want {
            t.Errorf("ModelType(%d).String() = %q, want %q", tt.mt, got, tt.want)
        }
    }
}
```

**Step 4: Run tests**

Run: `go test ./... -v -run TestModelType`
Expected: PASS

**Step 5: Commit**

```bash
git add types.go errors.go types_test.go
git commit -m "feat: add core types and error definitions"
```

---

### Task 3: Post-Processing (Sigmoid + Top-K)

**Files:**
- Create: `postprocess.go`
- Create: `postprocess_test.go`

**Step 1: Write failing tests**

Create `postprocess_test.go` with:
- `TestSigmoid` - verify sigmoid(0)=0.5, sigmoid(large)≈1.0, sigmoid(-large)≈0.0
- `TestTopKPredictions` - 10 scores, k=3, verify top 3 returned sorted descending
- `TestTopKPredictionsWithMinConfidence` - filter out low-confidence results
- `TestTopKPredictionsPreSigmoided` - BSG mode (no sigmoid applied)
- `TestTopKPredictionsEmptyInput` - empty scores returns empty result

Use concrete test data, e.g.:
```go
scores := []float32{0.1, 5.0, -2.0, 3.5, 0.0, 1.2, -1.0, 4.0, 2.5, -0.5}
labels := []string{"sp0", "sp1", "sp2", "sp3", "sp4", "sp5", "sp6", "sp7", "sp8", "sp9"}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./... -v -run TestSigmoid`
Expected: FAIL (function not defined)

**Step 3: Implement postprocess.go**

- `Sigmoid(x float32) float32` — `1.0 / (1.0 + float32(math.Exp(float64(-x))))`
- `TopKPredictions(scores []float32, labels []string, k int, minConf float32, preSigmoided bool) []Prediction`
  - Use `container/heap` min-heap of size k
  - Track indices of top-k raw scores
  - Apply sigmoid (or identity for preSigmoided) only to those k scores
  - Filter by minConf after activation
  - Sort by confidence descending

Reference: `../rust-birdnet-onnx/src/postprocess.rs` for the min-heap top-K algorithm.

**Step 4: Run tests**

Run: `go test ./... -v -run "TestSigmoid|TestTopK"`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add postprocess.go postprocess_test.go
git commit -m "feat: add sigmoid activation and top-K post-processing"
```

---

### Task 4: Label Loading

**Files:**
- Create: `labels.go`
- Create: `labels_test.go`
- Create: `testdata/labels_simple.txt`
- Create: `testdata/labels_v30.csv`
- Create: `testdata/labels_perch.json`

**Step 1: Create test fixture files**

`testdata/labels_simple.txt` (BirdNET v2.4 format):
```
Turdus merula_Common Blackbird
Parus major_Great Tit
Erithacus rubecula_European Robin
```

`testdata/labels_v30.csv` (BirdNET v3.0 format, semicolon-separated with header):
```
idx;sci_name;com_name;family;order
0;Turdus merula;Common Blackbird;Turdidae;Passeriformes
1;Parus major;Great Tit;Paridae;Passeriformes
2;Erithacus rubecula;European Robin;Muscicapidae;Passeriformes
```

`testdata/labels_perch.json`:
```json
["Turdus merula", "Parus major", "Erithacus rubecula"]
```

**Step 2: Write failing tests**

Create `labels_test.go` with:
- `TestLoadLabelsText` — loads simple text, returns 3 labels
- `TestLoadLabelsCSV` — loads CSV, picks `sci_name` column, returns 3 labels
- `TestLoadLabelsJSON` — loads JSON array, returns 3 labels
- `TestLoadLabelsFromReader` — test `LoadLabelsFromReader` with an `io.Reader`
- `TestLoadLabelsFileNotFound` — returns error

**Step 3: Run tests to verify they fail**

Run: `go test ./... -v -run TestLoadLabels`
Expected: FAIL

**Step 4: Implement labels.go**

- `LoadLabels(path string) ([]string, error)` — detect format by extension and content
- `LoadLabelsFromReader(r io.Reader, format string) ([]string, error)` — for `go:embed`
- Internal: `loadText`, `loadCSV` (smart column detection), `loadJSON`

Reference: `../rust-birdnet-onnx/src/labels.rs` for CSV column priority logic.

CSV column priority: `sci_name` > `com_name` > `species` > `name` > `label`. Use `encoding/csv` with auto-detection of delimiter (`,` vs `;`).

**Step 5: Run tests**

Run: `go test ./... -v -run TestLoadLabels`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add labels.go labels_test.go testdata/
git commit -m "feat: add label loading with text, CSV, and JSON support"
```

---

### Task 5: Model Auto-Detection

**Files:**
- Create: `detection.go`
- Create: `detection_test.go`

**Step 1: Write tests**

Create `detection_test.go` with:
- `TestDetectModelConfig` — table-driven test with mock input/output info structs
- Test cases for each model type: v2.4 (1 output, 144000 samples), v3.0 (2 outputs, 160000), Perch (4 outputs, 160000)
- Test case for dynamic dimensions falling back to output count
- Test case for unrecognizable shapes returning `ErrModelDetection`

Since `ort.GetInputOutputInfo` requires a real ONNX file, extract the core detection logic into a testable function:
```go
func detectFromShapes(inputs, outputs []tensorInfo) (ModelConfig, error)
```

Where `tensorInfo` is a lightweight struct matching `ort.InputOutputInfo` fields.

**Step 2: Run tests to verify they fail**

Run: `go test ./... -v -run TestDetect`
Expected: FAIL

**Step 3: Implement detection.go**

- `DetectModelType(modelPath string) (ModelConfig, error)` — calls `ort.GetInputOutputInfo`, delegates to `detectFromShapes`
- `detectFromShapes(inputs, outputs []tensorInfo) (ModelConfig, error)` — pure logic, testable without ONNX files

Detection algorithm (from design doc):
1. Get sample count from input tensor last dimension
2. Match sample count + output count to model type
3. Extract numSpecies from logits output last dimension
4. Extract embeddingDim from first output (v3.0/Perch)
5. Populate ModelConfig with sample rate, duration, etc.

Reference: `../rust-birdnet-onnx/src/detection.rs` for exact detection logic.

**Step 4: Run tests**

Run: `go test ./... -v -run TestDetect`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add detection.go detection_test.go
git commit -m "feat: add model auto-detection from ONNX tensor shapes"
```

---

### Task 6: Functional Options

**Files:**
- Create: `options.go`
- Create: `providers.go`

**Step 1: Implement options.go**

Define the internal config struct and all option functions:
```go
type classifierConfig struct {
    modelPath      string
    labels         []string
    labelsPath     string
    labelsReader   io.Reader
    labelsFormat   string
    modelType      *ModelType  // nil = auto-detect
    topK           int         // default: 10
    minConf        float32     // default: 0.0
    sessionOpts    *ort.SessionOptions
    providers      []providerConfig
}
```

Implement all `With*` functions from the design. Each validates its input and returns an error-returning closure.

**Step 2: Implement providers.go**

Execution provider helpers:
```go
type providerConfig struct {
    name    string
    setup   func(*ort.SessionOptions) error
}
```

Implement `WithCUDA`, `WithTensorRT`, `WithCoreML`, `WithDirectML`, `WithOpenVINO`, `WithExecutionProvider`.

Each creates an `ort.SessionOptions` if not already created, appends the provider.

**Step 3: Verify compilation**

Run: `go build ./...`
Expected: Compiles without errors

**Step 4: Commit**

```bash
git add options.go providers.go
git commit -m "feat: add functional options and execution provider helpers"
```

---

### Task 7: Classifier Core (NewClassifier + Predict)

**Files:**
- Create: `classifier.go`
- Create: `classifier_test.go`

This is the main integration point. It wires together detection, labels, options, post-processing, and ONNX session management.

**Step 1: Implement classifier.go**

`NewClassifier(opts ...Option) (*Classifier, error)`:
1. Apply all options to `classifierConfig`, validate required fields
2. Load labels (from path, reader, or direct slice)
3. Detect model type (or use override)
4. Check if batch dimension is dynamic
5. Create `ort.SessionOptions` with execution providers
6. Create pre-allocated input/output tensors:
   - Input: `[1, sampleCount]` float32
   - Output(s): based on model type (logits, embeddings)
7. Create `ort.AdvancedSession` with pre-allocated tensors
8. Validate label count == output dimension
9. Return configured `Classifier`

`Predict(ctx context.Context, segment []float32) (*PredictionResult, error)`:
1. Validate `len(segment) == config.SampleCount`
2. Lock mutex
3. Copy segment data into pre-allocated input tensor buffer
4. Create `RunOptions` with context monitoring goroutine (if ctx has deadline/cancel)
5. Run session
6. Extract output data from output tensor
7. Run `TopKPredictions` on logits output
8. Extract embeddings if applicable (v3.0/Perch)
9. Apply sigmoid to all raw scores for `RawScores` field
10. Unlock mutex, return result

`PredictBatch(ctx context.Context, segments [][]float32) ([]*PredictionResult, error)`:
- If `!dynamicBatch`: loop calling `Predict` for each segment
- If `dynamicBatch`: create batch tensors, run with `DynamicAdvancedSession`, split results

`Close() error`:
- Use `sync.Once` to destroy session, tensors, batch session

`Config() ModelConfig`, `Labels() []string` — simple getters.

Reference files:
- `../rust-birdnet-onnx/src/classifier.rs` — session creation, predict, batch logic
- `../birdnet-go/internal/birdnet/analyze.go` — Go patterns for tensor handling

**Step 2: Write classifier_test.go**

Tests that don't require real ONNX models:
- `TestNewClassifierMissingModelPath` — returns `ErrModelPath`
- `TestNewClassifierMissingLabels` — returns `ErrLabelsRequired`
- `TestPredictInputSizeValidation` — wrong segment size returns `InputSizeError`

Note: Full integration tests (Task 9) will test with real models.

**Step 3: Run tests**

Run: `go test ./... -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add classifier.go classifier_test.go
git commit -m "feat: add Classifier with Predict and PredictBatch"
```

---

### Task 8: Range Filter

**Files:**
- Create: `rangefilter.go`
- Create: `rangefilter_test.go`

**Step 1: Write failing tests**

Create `rangefilter_test.go`:
- `TestCalculateWeek` — table-driven: Jan 1→1, Jun 15→22, Dec 31→48
- `TestCalculateWeekBoundaries` — edge cases
- `TestNewRangeFilterValidation` — invalid path returns error

Reference: `../rust-birdnet-onnx/src/rangefilter.rs` for week calculation formula:
```
week = (month-1)*4 + (day-1)/7 + 1
```
Clamp to range [1, 48].

**Step 2: Run tests to verify they fail**

Run: `go test ./... -v -run "TestCalculateWeek|TestNewRangeFilter"`
Expected: FAIL

**Step 3: Implement rangefilter.go**

- `CalculateWeek(month, day int) float32`
- `NewRangeFilter(modelPath string, labels []string) (*RangeFilter, error)`:
  - Inspect meta model input/output shapes
  - Input: `[1, 3]` (lat, lon, week)
  - Output: `[1, numSpecies]`
  - Create pre-allocated tensors and `AdvancedSession`
- `GetSpeciesScores(lat, lon, week float32) ([]float32, error)`:
  - Validate coordinates and week
  - Lock mutex
  - Copy input data, run inference, copy output
  - Unlock and return
- `Close() error` — `sync.Once`, destroy session and tensors

**Step 4: Run tests**

Run: `go test ./... -v -run "TestCalculateWeek|TestNewRangeFilter"`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add rangefilter.go rangefilter_test.go
git commit -m "feat: add range filter for location/date species filtering"
```

---

### Task 9: Integration Tests

**Files:**
- Create: `integration_test.go`

These tests require real ONNX model files. Use build tags to skip when models aren't available.

**Step 1: Create integration test file**

```go
//go:build integration

package birdnet_test
```

Tests (each guarded by model file existence check via `testing.Short()` or env var):
- `TestBirdNetV24EndToEnd` — load v2.4 model + labels, predict silence (all zeros), verify returns results with low confidence
- `TestBirdNetV30EndToEnd` — load v3.0 model, verify embeddings are returned
- `TestPerchV2EndToEnd` — load Perch model, verify embeddings returned
- `TestBSGFinlandEndToEnd` — load BSG model with explicit `WithModelType`, verify no sigmoid applied
- `TestBatchPrediction` — predict 4 segments, verify 4 results
- `TestRangeFilterEndToEnd` — load meta model, get species scores for known location
- `TestContextCancellation` — cancel context before inference completes

Each test should:
1. Skip if model file not found: `testutil.SkipIfModelMissing(t, path)`
2. Initialize ONNX Runtime in `TestMain`
3. Create classifier, run prediction, verify result structure

**Step 2: Run unit tests (no integration)**

Run: `go test ./... -v`
Expected: ALL PASS (integration tests skipped)

**Step 3: Commit**

```bash
git add integration_test.go
git commit -m "feat: add integration tests for all model types"
```

---

### Task 10: Final Cleanup and Verification

**Step 1: Run all linters**

```bash
go vet ./...
```
Expected: No issues

**Step 2: Verify documentation**

```bash
go doc -all .
```
Expected: All exported types and functions have documentation

**Step 3: Run full test suite**

```bash
go test ./... -v -count=1
```
Expected: ALL PASS

**Step 4: Commit any cleanup**

```bash
git add -A
git commit -m "chore: final cleanup and documentation"
```
