package birdnet

import (
	"errors"
	"fmt"
	"io"

	ort "github.com/yalue/onnxruntime_go"
)

// Option configures a Classifier during construction.
type Option func(*classifierConfig) error

// classifierConfig holds all builder parameters.
type classifierConfig struct {
	modelPath    string
	labels       []string
	labelsPath   string
	labelsReader io.Reader
	labelsFormat string // "text", "csv", "json" — used with labelsReader
	modelType    *ModelType
	topK         int     // default: 10
	minConf      float32 // default: 0.0
	sessionOpts  *ort.SessionOptions
	providers    []providerConfig
}

// defaultConfig returns a classifierConfig with sensible defaults.
func defaultConfig() classifierConfig {
	return classifierConfig{
		topK:    10,
		minConf: 0.0,
	}
}

// WithModelPath sets the file system path to the ONNX model.
func WithModelPath(path string) Option {
	return func(c *classifierConfig) error {
		if path == "" {
			return errors.New("birdnet: model path must not be empty")
		}
		c.modelPath = path
		return nil
	}
}

// WithLabelsFromFile sets the file system path to a labels file.
func WithLabelsFromFile(path string) Option {
	return func(c *classifierConfig) error {
		if path == "" {
			return errors.New("birdnet: labels path must not be empty")
		}
		c.labelsPath = path
		return nil
	}
}

// WithLabelsFromReader provides labels via an io.Reader (useful for go:embed).
// The format parameter must be one of "text", "csv", or "json".
func WithLabelsFromReader(r io.Reader, format string) Option {
	return func(c *classifierConfig) error {
		if r == nil {
			return errors.New("birdnet: labels reader must not be nil")
		}
		switch format {
		case "text", "csv", "json":
			// valid
		default:
			return fmt.Errorf("birdnet: unsupported labels format %q (want \"text\", \"csv\", or \"json\")", format)
		}
		c.labelsReader = r
		c.labelsFormat = format
		return nil
	}
}

// WithLabels provides species labels directly as a string slice.
func WithLabels(labels []string) Option {
	return func(c *classifierConfig) error {
		if len(labels) == 0 {
			return errors.New("birdnet: labels slice must not be empty")
		}
		c.labels = labels
		return nil
	}
}

// WithModelType overrides automatic model type detection.
func WithModelType(mt ModelType) Option {
	return func(c *classifierConfig) error {
		c.modelType = &mt
		return nil
	}
}

// WithTopK sets the maximum number of predictions returned. Must be > 0.
func WithTopK(k int) Option {
	return func(c *classifierConfig) error {
		if k <= 0 {
			return fmt.Errorf("birdnet: topK must be > 0, got %d", k)
		}
		c.topK = k
		return nil
	}
}

// WithMinConfidence sets the minimum confidence threshold. Must be >= 0.
func WithMinConfidence(min float32) Option {
	return func(c *classifierConfig) error {
		if min < 0 {
			return fmt.Errorf("birdnet: minConfidence must be >= 0, got %f", min)
		}
		c.minConf = min
		return nil
	}
}

// WithSessionOptions provides custom ONNX Runtime session options.
// When set, provider options are applied to this instance instead of creating new session options.
func WithSessionOptions(opts *ort.SessionOptions) Option {
	return func(c *classifierConfig) error {
		if opts == nil {
			return errors.New("birdnet: session options must not be nil")
		}
		c.sessionOpts = opts
		return nil
	}
}
