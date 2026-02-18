// Command analyze-cli runs BirdNET inference on WAV files and prints predictions.
//
// Input requirements:
//   - 16-bit PCM WAV files only (8-bit, 24-bit, 32-bit, and float formats are not supported)
//   - Mono or stereo (stereo is mixed to mono automatically)
//   - Any sample rate (resampled to the model's expected rate if needed: 48 kHz for BirdNET v2.4, 32 kHz for v3.0/Perch)
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"strings"

	birdnet "github.com/tphakala/birdnet-onnx-go"
	resampler "github.com/tphakala/go-audio-resampler"
	"github.com/go-audio/wav"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	defaultTopK    = 10
	defaultMinConf = 0.01
	bitDepth16     = 16
	osDarwin       = "darwin"
)

func main() {
	modelPath := flag.String("model", "", "path to ONNX model file (required)")
	labelsPath := flag.String("labels", "", "path to labels file (required)")
	topK := flag.Int("topk", defaultTopK, "number of top predictions per segment")
	minConf := flag.Float64("min-conf", defaultMinConf, "minimum confidence threshold")
	overlap := flag.Float64("overlap", 0.0, "overlap between segments in seconds")
	ortLib := flag.String("ort-lib", "", "path to ONNX Runtime shared library (auto-detect if empty)")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <wav-file> [wav-file...]\n\nFlags:\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	if *modelPath == "" || *labelsPath == "" {
		flag.Usage()
		os.Exit(1)
	}
	wavFiles := flag.Args()
	if len(wavFiles) == 0 {
		flag.Usage()
		os.Exit(1)
	}

	// Initialize ONNX Runtime.
	libPath := *ortLib
	if libPath == "" {
		libPath = findORTLibrary()
	}
	if libPath != "" {
		ort.SetSharedLibraryPath(libPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("Failed to initialize ONNX Runtime: %v", err)
	}
	defer func() { _ = ort.DestroyEnvironment() }()

	// Create classifier.
	c, err := birdnet.NewClassifier(
		birdnet.WithModelPath(*modelPath),
		birdnet.WithLabelsFromFile(*labelsPath),
		birdnet.WithTopK(*topK),
		birdnet.WithMinConfidence(float32(*minConf)),
	)
	if err != nil {
		log.Fatalf("Failed to create classifier: %v", err) //nolint:gocritic // exitAfterDefer acceptable in CLI main
	}
	defer func() { _ = c.Close() }()

	cfg := c.Config()
	fmt.Printf("Model: %s  SampleRate: %d  Duration: %.1fs  Species: %d\n\n",
		cfg.ModelType, cfg.SampleRate, cfg.Duration, cfg.NumSpecies)

	overlapSamples := int(float64(cfg.SampleRate) * *overlap)

	ctx := context.Background()
	for _, path := range wavFiles {
		if err := processFile(ctx, c, cfg, path, overlapSamples); err != nil {
			log.Printf("Error processing %s: %v", path, err)
		}
	}
}

func processFile(ctx context.Context, c *birdnet.Classifier, cfg birdnet.ModelConfig, path string, overlapSamples int) error {
	fmt.Printf("=== %s ===\n", path)

	samples, sampleRate, err := loadWAV(path)
	if err != nil {
		return fmt.Errorf("loading WAV: %w", err)
	}
	fmt.Printf("Loaded %d samples at %d Hz (%.1fs)\n", len(samples), sampleRate, float64(len(samples))/float64(sampleRate))

	samples, err = resampleIfNeeded(samples, sampleRate, cfg.SampleRate)
	if err != nil {
		return fmt.Errorf("resampling: %w", err)
	}

	segments := splitSegments(samples, cfg.SampleCount, overlapSamples)
	fmt.Printf("Split into %d segment(s)\n\n", len(segments))

	segDuration := cfg.Duration
	stepSamples := cfg.SampleCount - overlapSamples
	stepDuration := float64(stepSamples) / float64(cfg.SampleRate)

	for i, seg := range segments {
		startSec := float64(i) * stepDuration
		endSec := startSec + segDuration

		result, err := c.Predict(ctx, seg)
		if err != nil {
			return fmt.Errorf("predict segment %d: %w", i, err)
		}

		fmt.Printf("segment %.1fs-%.1fs:\n", startSec, endSec)
		for j, pred := range result.Predictions {
			fmt.Printf("  %2d. %-50s %.4f\n", j+1, pred.Species, pred.Confidence)
		}
		fmt.Println()
	}

	return nil
}

// loadWAV reads a 16-bit PCM WAV file and returns mono float32 samples
// normalized to [-1.0, 1.0]. Stereo files are mixed to mono by averaging
// channels. Other bit depths and float formats are not supported.
func loadWAV(path string) (samples []float32, sampleRate int, err error) {
	f, err := os.Open(path) //nolint:gosec // path comes from CLI flag, not user-controlled input
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = f.Close() }()

	dec := wav.NewDecoder(f)

	buf, err := dec.FullPCMBuffer()
	if err != nil {
		return nil, 0, fmt.Errorf("reading PCM data: %w", err)
	}

	if dec.BitDepth != bitDepth16 {
		return nil, 0, fmt.Errorf("%s: unsupported bit depth %d (only 16-bit PCM is supported)", path, dec.BitDepth)
	}

	sampleRate = int(dec.SampleRate)
	numChannels := int(dec.NumChans)

	const scale = 1.0 / 32768.0

	intData := buf.Data
	numFrames := len(intData) / numChannels

	samples = make([]float32, numFrames)
	if numChannels == 1 {
		for i := range numFrames {
			samples[i] = float32(intData[i]) * scale
		}
	} else {
		// Mix to mono by averaging channels.
		for i := range numFrames {
			var sum float32
			for ch := range numChannels {
				sum += float32(intData[i*numChannels+ch])
			}
			samples[i] = (sum / float32(numChannels)) * scale
		}
	}

	return samples, sampleRate, nil
}

// resampleIfNeeded resamples audio if the input rate differs from the target rate.
func resampleIfNeeded(samples []float32, inRate, outRate int) ([]float32, error) {
	if inRate == outRate {
		return samples, nil
	}
	fmt.Printf("Resampling from %d Hz to %d Hz\n", inRate, outRate)

	r, err := resampler.NewSimple(float64(inRate), float64(outRate))
	if err != nil {
		return nil, fmt.Errorf("creating resampler: %w", err)
	}

	out, err := r.ProcessFloat32(samples)
	if err != nil {
		return nil, fmt.Errorf("resampling: %w", err)
	}

	return out, nil
}

// splitSegments divides audio into fixed-size segments with optional overlap.
// The last segment is zero-padded if shorter than segmentSize.
func splitSegments(samples []float32, segmentSize, overlapSize int) [][]float32 {
	if len(samples) == 0 {
		return nil
	}

	step := segmentSize - overlapSize
	if step <= 0 {
		step = segmentSize
	}

	var segments [][]float32
	for offset := 0; offset < len(samples); offset += step {
		seg := make([]float32, segmentSize)
		n := copy(seg, samples[offset:])
		_ = n // remaining zeros from make() act as padding
		segments = append(segments, seg)
	}

	return segments
}

// findORTLibrary searches common paths for the ONNX Runtime shared library.
func findORTLibrary() string {
	var candidates []string

	switch runtime.GOOS {
	case osDarwin:
		candidates = []string{
			"/usr/local/lib/libonnxruntime.dylib",
			"/opt/homebrew/lib/libonnxruntime.dylib",
		}
	case "linux":
		candidates = []string{
			"/usr/lib/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
			"/usr/lib/aarch64-linux-gnu/libonnxruntime.so",
		}
	case "windows":
		candidates = []string{
			`C:\Program Files\onnxruntime\lib\onnxruntime.dll`,
		}
	}

	// Also check LD_LIBRARY_PATH / DYLD_LIBRARY_PATH entries.
	envVar := "LD_LIBRARY_PATH"
	if runtime.GOOS == osDarwin {
		envVar = "DYLD_LIBRARY_PATH"
	}
	if paths := os.Getenv(envVar); paths != "" {
		for dir := range strings.SplitSeq(paths, string(os.PathListSeparator)) {
			switch runtime.GOOS {
			case osDarwin:
				candidates = append(candidates, dir+"/libonnxruntime.dylib")
			case "linux":
				candidates = append(candidates, dir+"/libonnxruntime.so")
			}
		}
	}

	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	return ""
}

