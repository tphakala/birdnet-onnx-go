package birdnet

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// providerConfig holds setup instructions for an ONNX Runtime execution provider.
type providerConfig struct {
	name  string
	setup func(*ort.SessionOptions) error
}

// WithCUDA appends the CUDA execution provider with the given device ID.
func WithCUDA(deviceID int) Option {
	return func(c *classifierConfig) error {
		c.providers = append(c.providers, providerConfig{
			name: "CUDA",
			setup: func(opts *ort.SessionOptions) error {
				cudaOpts, err := ort.NewCUDAProviderOptions()
				if err != nil {
					return fmt.Errorf("birdnet: create CUDA provider options: %w", err)
				}
				if err := cudaOpts.Update(map[string]string{
					"device_id": fmt.Sprintf("%d", deviceID),
				}); err != nil {
					return fmt.Errorf("birdnet: update CUDA provider options: %w", err)
				}
				return opts.AppendExecutionProviderCUDA(cudaOpts)
			},
		})
		return nil
	}
}

// WithCUDAOptions appends the CUDA execution provider with full control over options.
func WithCUDAOptions(options map[string]string) Option {
	return func(c *classifierConfig) error {
		c.providers = append(c.providers, providerConfig{
			name: "CUDA",
			setup: func(opts *ort.SessionOptions) error {
				cudaOpts, err := ort.NewCUDAProviderOptions()
				if err != nil {
					return fmt.Errorf("birdnet: create CUDA provider options: %w", err)
				}
				if err := cudaOpts.Update(options); err != nil {
					return fmt.Errorf("birdnet: update CUDA provider options: %w", err)
				}
				return opts.AppendExecutionProviderCUDA(cudaOpts)
			},
		})
		return nil
	}
}

// WithTensorRT appends the TensorRT execution provider with the given device ID.
func WithTensorRT(deviceID int) Option {
	return func(c *classifierConfig) error {
		c.providers = append(c.providers, providerConfig{
			name: "TensorRT",
			setup: func(opts *ort.SessionOptions) error {
				trtOpts, err := ort.NewTensorRTProviderOptions()
				if err != nil {
					return fmt.Errorf("birdnet: create TensorRT provider options: %w", err)
				}
				if err := trtOpts.Update(map[string]string{
					"device_id": fmt.Sprintf("%d", deviceID),
				}); err != nil {
					return fmt.Errorf("birdnet: update TensorRT provider options: %w", err)
				}
				return opts.AppendExecutionProviderTensorRT(trtOpts)
			},
		})
		return nil
	}
}

// WithTensorRTOptions appends the TensorRT execution provider with full control over options.
func WithTensorRTOptions(options map[string]string) Option {
	return func(c *classifierConfig) error {
		c.providers = append(c.providers, providerConfig{
			name: "TensorRT",
			setup: func(opts *ort.SessionOptions) error {
				trtOpts, err := ort.NewTensorRTProviderOptions()
				if err != nil {
					return fmt.Errorf("birdnet: create TensorRT provider options: %w", err)
				}
				if err := trtOpts.Update(options); err != nil {
					return fmt.Errorf("birdnet: update TensorRT provider options: %w", err)
				}
				return opts.AppendExecutionProviderTensorRT(trtOpts)
			},
		})
		return nil
	}
}

// WithCoreML appends the CoreML execution provider with default flags (0).
func WithCoreML() Option {
	return func(c *classifierConfig) error {
		c.providers = append(c.providers, providerConfig{
			name: "CoreML",
			setup: func(opts *ort.SessionOptions) error {
				return opts.AppendExecutionProviderCoreML(0)
			},
		})
		return nil
	}
}

// WithDirectML appends the DirectML execution provider with the given device ID.
func WithDirectML(deviceID int) Option {
	return func(c *classifierConfig) error {
		c.providers = append(c.providers, providerConfig{
			name: "DirectML",
			setup: func(opts *ort.SessionOptions) error {
				return opts.AppendExecutionProviderDirectML(deviceID)
			},
		})
		return nil
	}
}

// WithOpenVINO appends the OpenVINO execution provider with empty options.
func WithOpenVINO() Option {
	return func(c *classifierConfig) error {
		c.providers = append(c.providers, providerConfig{
			name: "OpenVINO",
			setup: func(opts *ort.SessionOptions) error {
				return opts.AppendExecutionProviderOpenVINO(map[string]string{})
			},
		})
		return nil
	}
}

// WithExecutionProvider appends a generic execution provider by name with the given options.
func WithExecutionProvider(name string, options map[string]string) Option {
	return func(c *classifierConfig) error {
		if name == "" {
			return fmt.Errorf("birdnet: execution provider name must not be empty")
		}
		c.providers = append(c.providers, providerConfig{
			name: name,
			setup: func(opts *ort.SessionOptions) error {
				return opts.AppendExecutionProvider(name, options)
			},
		})
		return nil
	}
}
