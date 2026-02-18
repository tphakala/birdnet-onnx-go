# birdnet-onnx-go Design Document

## Overview

A Go library for bird species detection using ONNX Runtime. Supports BirdNET v2.4, BirdNET v3.0, Google Perch v2, and BSG Finland v4.4 models. Uses [onnxruntime_go](https://github.com/yalue/onnxruntime_go) for the ONNX Runtime interface.

**Module path:** `github.com/tphakala/birdnet-onnx-go`
**Go version:** 1.26
**Reference implementation:** [rust-birdnet-onnx](../../../rust-birdnet-onnx/)

## Goals

- Clean, idiomatic Go 1.26 API with functional options pattern
- Support all 4 model families with automatic model type detection from ONNX tensor shapes
- Batch inference support for processing multiple audio segments in one call
- All ONNX Runtime execution providers (CUDA, TensorRT, CoreML, DirectML, OpenVINO, etc.)
- Range filter (meta model) for location/date-based species filtering
- External labels only (no embedded data) - labels provided as file paths or string slices
- Thread-safe with pre-allocated buffer reuse for streaming workloads

## Non-Goals (initial version)

- BSG Finland post-processing (calibration CSV, species distribution model)
- Audio preprocessing (resampling, FFT) - callers provide raw float32 audio
- Embedded label files
- Training support

## Supported Models

| Model | Sample Rate | Duration | Samples | Outputs | Embeddings | Activation |
|-------|------------|----------|---------|---------|------------|------------|
| BirdNET v2.4 | 48 kHz | 3.0s | 144,000 | 1 (logits) | None | sigmoid |
| BirdNET v3.0 | 32 kHz | 5.0s | 160,000 | 2 (embeddings + logits) | 1,280-dim | sigmoid |
| Google Perch v2 | 32 kHz | 5.0s | 160,000 | 4 (embeddings + logits + others) | variable | sigmoid |
| BSG Finland v4.4 | 48 kHz | 3.0s | 144,000 | 1 (scores) | None | identity (pre-sigmoided) |

## Architecture

### Package Structure

Single package `birdnet` with focused source files:

```
github.com/tphakala/birdnet-onnx-go/
├── go.mod
├── go.sum
├── doc.go              # Package documentation
├── classifier.go       # Classifier struct, NewClassifier, Predict, PredictBatch, Close
├── detection.go        # Auto-detect ModelType from ONNX input/output tensor shapes
├── labels.go           # Load labels from text, CSV, JSON files; smart column detection
├── postprocess.go      # Top-K selection (min-heap), sigmoid activation
├── rangefilter.go      # Range filter meta model (location/date species filtering)
├── types.go            # ModelType, ModelConfig, Prediction, PredictionResult
├── errors.go           # Sentinel errors and typed error values
├── options.go          # Functional options for NewClassifier, InferenceOptions
├── providers.go        # Execution provider option helpers (WithCUDA, WithCoreML, etc.)
├── classifier_test.go  # Unit tests for classifier
├── detection_test.go   # Unit tests for model detection
├── labels_test.go      # Unit tests for label loading
├── postprocess_test.go # Unit tests for top-K and sigmoid
├── rangefilter_test.go # Unit tests for range filter
└── testdata/           # Test fixtures (small ONNX models, label files)
```

### Core Types (`types.go`)

```go
// ModelType identifies which bird detection model is loaded.
type ModelType int

const (
    ModelTypeBirdNetV24 ModelType = iota
    ModelTypeBirdNetV30
    ModelTypePerchV2
    ModelTypeBSGFinland
)

// ModelConfig holds the detected or overridden model configuration.
type ModelConfig struct {
    ModelType    ModelType
    SampleRate   int     // 48000 or 32000
    Duration     float64 // 3.0 or 5.0 seconds
    SampleCount  int     // 144000 or 160000
    NumSpecies   int     // from model output shape
    EmbeddingDim int     // 0 for v2.4/BSG, 1280 for v3.0, variable for Perch
    PreSigmoided bool    // true for BSG only (scores already in [0,1])
}

// Prediction is a single species detection result.
type Prediction struct {
    Species    string  // Label text (e.g. "Turdus merula_Common Blackbird")
    Confidence float32 // 0.0-1.0 after sigmoid/identity
    Index      int     // Position in the label array
}

// PredictionResult contains inference output for one audio segment.
type PredictionResult struct {
    ModelType   ModelType
    Predictions []Prediction // Top-K, sorted by confidence descending
    Embeddings  []float32    // nil for v2.4 and BSG
    RawScores   []float32    // All species scores after activation
}
```

### Classifier (`classifier.go`)

```go
// Classifier performs bird species detection using an ONNX model.
type Classifier struct {
    session      *ort.AdvancedSession  // ONNX Runtime session (pre-allocated tensors)
    config       ModelConfig
    labels       []string
    topK         int
    minConf      float32
    dynamicBatch bool                  // true if model supports batch dim > 1
    mu           sync.Mutex
    closeOnce    sync.Once
    // Pre-allocated buffers for single-segment inference
    inputData    []float32
    outputData   []float32
    inputTensor  *ort.Tensor[float32]
    outputTensor *ort.Tensor[float32]
    // Embedding output tensor (nil for v2.4/BSG)
    embOutputTensor *ort.Tensor[float32]
    // Batch session (created lazily on first PredictBatch call, nil if fixed batch)
    batchSession *ort.DynamicAdvancedSession
}

// NewClassifier creates a new Classifier with the given options.
// It loads the ONNX model, auto-detects the model type (unless overridden),
// loads labels, validates label count against model output, and creates
// the ONNX Runtime session with pre-allocated tensors.
func NewClassifier(opts ...Option) (*Classifier, error)

// Predict runs inference on a single audio segment.
// The segment must be []float32 with length == config.SampleCount.
// Use context.Background() for no timeout/cancellation.
func (c *Classifier) Predict(ctx context.Context, segment []float32) (*PredictionResult, error)

// PredictBatch runs inference on multiple audio segments in one call.
// All segments must have length == config.SampleCount.
// Returns one PredictionResult per segment.
func (c *Classifier) PredictBatch(ctx context.Context, segments [][]float32) ([]*PredictionResult, error)

// Config returns the detected model configuration.
func (c *Classifier) Config() ModelConfig

// Labels returns the loaded species labels.
func (c *Classifier) Labels() []string

// Close releases ONNX Runtime resources.
func (c *Classifier) Close() error
```

#### Batch Implementation Strategy

During `NewClassifier`, the model's input tensor shape is inspected:
- If the batch dimension is dynamic (or > 1), `dynamicBatch = true`
- If the batch dimension is fixed to 1, `dynamicBatch = false`

`PredictBatch` behavior:
- If `dynamicBatch == true`: Uses `DynamicAdvancedSession` with batch-sized tensors `[batchSize, sampleCount]`. A separate `batchSession` is created lazily on first call and reused. New tensors are allocated per call since batch sizes may vary.
- If `dynamicBatch == false`: Falls back to sequential processing - calls `Predict` in a loop for each segment. This is transparent to the caller.
- For single-segment `Predict`, pre-allocated tensors via `AdvancedSession` avoid allocation.

### Model Auto-Detection (`detection.go`)

Mirrors the Rust detection logic using `ort.GetInputOutputInfo()`:

```go
// DetectModelType inspects ONNX model tensor shapes to determine the model type.
func DetectModelType(modelPath string) (ModelConfig, error)
```

**Detection algorithm:**
1. Read input/output tensor info from ONNX file
2. Check input tensor dimensions for sample count:
   - 144,000 samples + 1 output → BirdNET v2.4
   - 160,000 samples + 2 outputs → BirdNET v3.0
   - 160,000 samples + 4 outputs → Perch v2
3. If input has dynamic dimensions, infer from output count:
   - 1 output → BirdNET v2.4
   - 2 outputs → BirdNET v3.0
   - 4 outputs → Perch v2
4. BSG Finland has same shape as v2.4 → requires explicit `WithModelType()` override
5. Extract `numSpecies` from last dimension of the logits output tensor
6. Extract `embeddingDim` from first output tensor (v3.0/Perch only)

### Label Loading (`labels.go`)

```go
// LoadLabels reads species labels from a file, auto-detecting format.
func LoadLabels(path string) ([]string, error)

// LoadLabelsFromCSV reads labels from CSV with smart column detection.
func LoadLabelsFromCSV(path string) ([]string, error)
```

**Format detection:**
- `.json` extension → parse as JSON array of strings or objects
- `.csv` extension or contains `;`/`,` in first line → CSV with column priority: `sci_name` > `com_name` > `species` > `name` > `label`
- Otherwise → plain text, one label per line

### Post-Processing (`postprocess.go`)

```go
// TopKPredictions extracts the top-K predictions from raw model output.
// Uses a min-heap for O(n log k) selection.
// Applies sigmoid activation unless preSigmoided is true.
func TopKPredictions(scores []float32, labels []string, k int, minConf float32, preSigmoided bool) []Prediction

// Sigmoid computes 1/(1 + exp(-x)).
func Sigmoid(x float32) float32
```

**Key detail:** Sigmoid is only applied to the top-K scores (not all 6,522), saving computation.

### Range Filter (`rangefilter.go`)

```go
// RangeFilter uses a BirdNET meta model to filter species by location and date.
type RangeFilter struct {
    session      *ort.AdvancedSession
    labels       []string
    mu           sync.Mutex
    inputTensor  *ort.Tensor[float32]
    outputTensor *ort.Tensor[float32]
}

// NewRangeFilter creates a range filter from a meta model ONNX file.
func NewRangeFilter(modelPath string, labels []string) (*RangeFilter, error)

// GetSpeciesScores returns occurrence scores for all species at a location/date.
// lat: -90 to 90, lon: -180 to 180, week: 1-48.
func (rf *RangeFilter) GetSpeciesScores(lat, lon, week float32) ([]float32, error)

// CalculateWeek converts month (1-12) and day (1-31) to BirdNET's 48-week calendar.
func CalculateWeek(month, day int) float32

// Close releases ONNX Runtime resources.
func (rf *RangeFilter) Close() error
```

### Context-Based Cancellation

Inference methods accept `context.Context` as their first argument, following Go conventions:

```go
func (c *Classifier) Predict(ctx context.Context, segment []float32) (*PredictionResult, error)
func (c *Classifier) PredictBatch(ctx context.Context, segments [][]float32) ([]*PredictionResult, error)
```

Callers use standard `context.WithTimeout` or `context.WithCancel` for timeout/cancellation.
Internally, a monitoring goroutine watches `ctx.Done()` and calls `ort.RunOptions.Terminate()`.
Pass `context.Background()` for no timeout/cancellation.

### Functional Options (`options.go`)

```go
type Option func(*classifierConfig) error

// Required
func WithModelPath(path string) Option
func WithLabelsFromFile(path string) Option
func WithLabelsFromReader(r io.Reader) Option  // For go:embed or other sources
func WithLabels(labels []string) Option

// Optional
func WithModelType(mt ModelType) Option    // Override auto-detection
func WithTopK(k int) Option               // Default: 10
func WithMinConfidence(min float32) Option // Default: 0.0
func WithSessionOptions(opts *ort.SessionOptions) Option // Full control

// Execution providers
func WithCUDA(deviceID int) Option
func WithCUDAOptions(opts map[string]string) Option
func WithTensorRT(deviceID int) Option
func WithTensorRTOptions(opts map[string]string) Option
func WithCoreML() Option
func WithDirectML(deviceID int) Option
func WithOpenVINO() Option
func WithExecutionProvider(name string, opts map[string]string) Option
```

### Error Handling (`errors.go`)

```go
var (
    ErrInputSize      = errors.New("birdnet: input segment size mismatch")
    ErrBatchInputSize = errors.New("birdnet: batch segment size mismatch")
    ErrModelDetection = errors.New("birdnet: unable to detect model type from ONNX shapes")
    ErrLabelCount     = errors.New("birdnet: label count does not match model output dimension")
    ErrModelPath      = errors.New("birdnet: model path is required")
    ErrLabelsRequired = errors.New("birdnet: labels are required")
    // Context cancellation/timeout uses standard context.Canceled / context.DeadlineExceeded
    ErrInvalidCoords  = errors.New("birdnet: invalid coordinates")
    ErrInvalidWeek    = errors.New("birdnet: week must be between 1 and 48")
)

// InputSizeError provides details about size mismatches.
type InputSizeError struct {
    Expected int
    Got      int
}
```

Errors use `fmt.Errorf("...: %w", sentinel)` wrapping so callers can use `errors.Is()`.

### Thread Safety Model

- `Classifier.Predict` and `PredictBatch` acquire `sync.Mutex` before touching session/buffers
- Pre-allocated input/output tensors and score buffers are reused under the lock
- `RangeFilter` similarly mutex-protected
- `Close()` is idempotent via `sync.Once` - safe to call multiple times
- **Concurrency note:** A single `Classifier` serializes inference. For concurrent high-throughput processing, create a pool of `Classifier` instances (one per goroutine or worker).

### ONNX Runtime Lifecycle

The library does NOT call `ort.InitializeEnvironment()` or `ort.DestroyEnvironment()`. The caller is responsible for ONNX Runtime lifecycle management. This keeps the library composable - multiple libraries can share one environment.

```go
// Caller's responsibility:
ort.SetSharedLibraryPath("/path/to/onnxruntime.so")
ort.InitializeEnvironment()
defer ort.DestroyEnvironment()

// Then use the library:
c, _ := birdnet.NewClassifier(...)
```

## Usage Examples

### Basic Inference

```go
import (
    ort "github.com/yalue/onnxruntime_go"
    "github.com/tphakala/birdnet-onnx-go"
)

// Initialize ONNX Runtime (caller's responsibility)
ort.SetSharedLibraryPath("./onnxruntime.so")
ort.InitializeEnvironment()
defer ort.DestroyEnvironment()

// Create classifier
c, err := birdnet.NewClassifier(
    birdnet.WithModelPath("BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx"),
    birdnet.WithLabelsFromFile("BirdNET_GLOBAL_6K_V2.4_Labels_en.txt"),
    birdnet.WithTopK(5),
    birdnet.WithMinConfidence(0.1),
)
if err != nil {
    log.Fatal(err)
}
defer c.Close()

// Predict (audio must be 48kHz mono, 3 seconds = 144,000 samples)
result, err := c.Predict(context.Background(), audioSegment)
if err != nil {
    log.Fatal(err)
}

for _, p := range result.Predictions {
    fmt.Printf("%s: %.1f%%\n", p.Species, p.Confidence*100)
}
```

### GPU Inference with Timeout

```go
c, err := birdnet.NewClassifier(
    birdnet.WithModelPath("model.onnx"),
    birdnet.WithLabelsFromFile("labels.txt"),
    birdnet.WithCUDA(0),
)

ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()
result, err := c.Predict(ctx, segment)
```

### Batch Inference

```go
segments := [][]float32{segment1, segment2, segment3}
results, err := c.PredictBatch(context.Background(), segments)
for i, r := range results {
    fmt.Printf("Segment %d: %s (%.1f%%)\n", i, r.Predictions[0].Species, r.Predictions[0].Confidence*100)
}
```

### Range Filter

```go
rf, err := birdnet.NewRangeFilter("meta_model.onnx", c.Labels())
defer rf.Close()

week := birdnet.CalculateWeek(6, 15) // June 15
scores, err := rf.GetSpeciesScores(60.17, 24.94, week) // Helsinki
```
