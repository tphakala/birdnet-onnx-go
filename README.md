# birdnet-onnx-go

Go library for bird species detection using [BirdNET](https://birdnet.cornell.edu/) ONNX models via [ONNX Runtime](https://onnxruntime.ai/).

Supports BirdNET v2.4, BirdNET v3.0, Google Perch v2, and BSG Finland v4.4 with automatic model type detection from ONNX tensor shapes.

## Features

- **Automatic model detection** from ONNX input/output tensor shapes
- **Multiple model families** with different sample rates, durations, and output formats
- **Thread-safe** inference with pre-allocated tensor reuse
- **Batch prediction** with automatic dynamic batching when the model supports it
- **Geographic range filtering** using a separate ONNX model to filter species by location and time of year
- **Flexible label loading** with auto-detection of text, CSV, and JSON formats
- **Hardware acceleration** via CUDA, TensorRT, CoreML, DirectML, and OpenVINO execution providers
- **Functional options** API for clean, extensible configuration

## Supported Models

| Model | Sample Rate | Segment Duration | Embeddings | Species |
|-------|------------|-----------------|------------|---------|
| BirdNET v2.4 | 48 kHz | 3.0s | No | ~6500 |
| BirdNET v3.0 | 32 kHz | 5.0s | Yes (1280-dim) | ~6500 |
| Google Perch v2 | 32 kHz | 5.0s | Yes (variable) | ~10000 |
| BSG Finland v4.4 | 48 kHz | 3.0s | No | ~500 |

## Requirements

- Go 1.25+
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) shared library (1.24+)
- A BirdNET ONNX model file and corresponding labels file

## Installation

```bash
go get github.com/tphakala/birdnet-onnx-go
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    birdnet "github.com/tphakala/birdnet-onnx-go"
    ort "github.com/yalue/onnxruntime_go"
)

func main() {
    // Initialize ONNX Runtime (caller responsibility)
    ort.SetSharedLibraryPath("/path/to/libonnxruntime.so")
    if err := ort.InitializeEnvironment(); err != nil {
        log.Fatal(err)
    }
    defer ort.DestroyEnvironment()

    // Create classifier
    c, err := birdnet.NewClassifier(
        birdnet.WithModelPath("birdnet.onnx"),
        birdnet.WithLabelsFromFile("labels.txt"),
        birdnet.WithTopK(10),
        birdnet.WithMinConfidence(0.1),
    )
    if err != nil {
        log.Fatal(err)
    }
    defer c.Close()

    // Check model configuration
    cfg := c.Config()
    fmt.Printf("Model: %s  SampleRate: %d  Duration: %.1fs  Species: %d\n",
        cfg.ModelType, cfg.SampleRate, cfg.Duration, cfg.NumSpecies)

    // Prepare audio segment (must have exactly cfg.SampleCount float32 samples)
    segment := make([]float32, cfg.SampleCount)
    // ... fill segment with audio data ...

    // Run inference
    result, err := c.Predict(context.Background(), segment)
    if err != nil {
        log.Fatal(err)
    }

    for _, pred := range result.Predictions {
        fmt.Printf("%-50s  %.4f\n", pred.Species, pred.Confidence)
    }
}
```

## API

### Classifier

The central type. Thread-safe for concurrent use.

```go
// Create with functional options
c, err := birdnet.NewClassifier(opts ...Option)

// Single segment inference
result, err := c.Predict(ctx, segment)

// Batch inference (auto-detects dynamic batching support)
results, err := c.PredictBatch(ctx, segments)

// Inspect detected model configuration
cfg := c.Config()

// Get a copy of the loaded labels
labels := c.Labels()

// Release ONNX resources (idempotent, safe to call multiple times)
c.Close()
```

### Options

#### Required

```go
birdnet.WithModelPath("model.onnx")           // ONNX model file
birdnet.WithLabelsFromFile("labels.txt")       // Load labels from file (auto-detects format)
```

#### Label Loading Alternatives

```go
birdnet.WithLabelsFromReader(r, "text")        // From io.Reader (useful with go:embed)
birdnet.WithLabels([]string{"species1", ...})  // Provide labels directly
```

Supported label formats: `"text"` (one per line), `"csv"` (with smart column/delimiter detection), `"json"` (string array).

#### Inference Tuning

```go
birdnet.WithTopK(10)                // Max predictions per inference (default: 10)
birdnet.WithMinConfidence(0.1)      // Confidence threshold (default: 0.0)
birdnet.WithModelType(mt)           // Override automatic model detection
```

#### Hardware Acceleration

```go
birdnet.WithCUDA(0)                            // CUDA with device ID
birdnet.WithCUDAOptions(map[string]string{})   // CUDA with custom options
birdnet.WithTensorRT(0)                        // TensorRT
birdnet.WithCoreML()                           // CoreML (macOS)
birdnet.WithDirectML(0)                        // DirectML (Windows)
birdnet.WithOpenVINO()                         // OpenVINO
birdnet.WithSessionOptions(opts)               // Custom ONNX session options
```

### Prediction Results

```go
result.ModelType    // Detected model type
result.Predictions  // []Prediction sorted by confidence (descending)
result.Embeddings   // []float32 (v3.0/Perch only, nil for v2.4/BSG)
result.RawScores    // []float32 scores for all species
```

Each `Prediction` contains:

```go
pred.Species     // Label string (e.g. "Glaucidium passerinum_Eurasian Pygmy-Owl")
pred.Confidence  // Float32 in [0.0, 1.0]
pred.Index       // Position in the labels array
```

### Range Filter

Geographic and temporal species filtering using a separate ONNX model.

```go
rf, err := birdnet.NewRangeFilter("range_model.onnx", labels)
defer rf.Close()

// Get species presence scores for a location and week
week := birdnet.CalculateWeek(3, 15) // March 15
scores, err := rf.GetSpeciesScores(60.17, 24.94, week) // Helsinki
```

### Model Detection

Detect model type without creating a full classifier:

```go
cfg, err := birdnet.DetectModelType("model.onnx")
fmt.Println(cfg.ModelType, cfg.SampleRate, cfg.NumSpecies)
```

### Label Loading

Standalone label loading with automatic format detection:

```go
labels, err := birdnet.LoadLabels("labels.txt")          // Auto-detect format
labels, err := birdnet.LoadLabelsFromReader(r, "csv")     // Explicit format
```

CSV loading features smart delimiter detection (`,` or `;`), header row recognition, and column priority selection (`sci_name` > `com_name` > `species` > `name` > `label`).

## CLI Tool

`analyze-cli` is a proof-of-concept command-line tool for running BirdNET inference on WAV files.

### Build

```bash
go build -o analyze-cli ./cmd/analyze-cli/
```

### Usage

```bash
analyze-cli -model birdnet.onnx -labels labels.txt [flags] file.wav [file2.wav ...]
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-model` | (required) | Path to ONNX model file |
| `-labels` | (required) | Path to labels file |
| `-topk` | 10 | Number of top predictions per segment |
| `-min-conf` | 0.01 | Minimum confidence threshold |
| `-overlap` | 0.0 | Overlap between segments in seconds |
| `-ort-lib` | (auto-detect) | Path to ONNX Runtime shared library |

### Input Requirements

- 16-bit PCM WAV files only
- Mono or stereo (stereo is mixed to mono automatically)
- Any sample rate (resampled to the model's expected rate if needed)

### Example

```
$ analyze-cli -model birdnet.onnx -labels labels.txt -topk 3 -min-conf 0.1 recording.wav
Model: BirdNET v2.4  SampleRate: 48000  Duration: 3.0s  Species: 6522

=== recording.wav ===
Loaded 9648000 samples at 48000 Hz (201.0s)
Split into 67 segment(s)

segment 0.0s-3.0s:
   1. Glaucidium passerinum_Eurasian Pygmy-Owl           0.9237

segment 3.0s-6.0s:
   1. Glaucidium passerinum_Eurasian Pygmy-Owl           0.9849
```

## License

See [LICENSE](LICENSE) for details.
