package birdnet

import (
	"context"
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// Classifier wraps an ONNX Runtime session for bird species classification.
// It is safe for concurrent use; Predict and PredictBatch acquire a mutex
// before copying data into pre-allocated tensors and running inference.
type Classifier struct {
	session      *ort.AdvancedSession
	config       ModelConfig
	labels       []string
	topK         int
	minConf      float32
	dynamicBatch bool
	mu           sync.Mutex
	closeOnce    sync.Once

	// Pre-allocated for single-segment inference.
	inputTensor   *ort.Tensor[float32]
	outputTensors []*ort.Tensor[float32] // 1 for v2.4/BSG, 2 for v3.0, 4 for Perch

	// Batch session (lazily created).
	batchSession *ort.DynamicAdvancedSession

	// Session options (kept for batch session creation).
	sessionOpts *ort.SessionOptions
	modelPath   string
	inputNames  []string
	outputNames []string
}

// NewClassifier creates a new Classifier with the given options.
// At minimum, a model path and labels must be provided.
func NewClassifier(opts ...Option) (*Classifier, error) {
	cfg := defaultConfig()
	for _, o := range opts {
		if err := o(&cfg); err != nil {
			return nil, fmt.Errorf("birdnet: applying option: %w", err)
		}
	}

	// Validate required fields.
	if cfg.modelPath == "" {
		return nil, ErrModelPath
	}

	// Load labels from one of three sources.
	var labels []string
	var err error
	switch {
	case len(cfg.labels) > 0:
		labels = cfg.labels
	case cfg.labelsPath != "":
		labels, err = LoadLabels(cfg.labelsPath)
		if err != nil {
			return nil, fmt.Errorf("birdnet: loading labels from file: %w", err)
		}
	case cfg.labelsReader != nil:
		labels, err = LoadLabelsFromReader(cfg.labelsReader, cfg.labelsFormat)
		if err != nil {
			return nil, fmt.Errorf("birdnet: loading labels from reader: %w", err)
		}
	default:
		return nil, ErrLabelsRequired
	}

	// Detect model type.
	detectedConfig, err := DetectModelType(cfg.modelPath)
	if err != nil {
		return nil, fmt.Errorf("birdnet: detecting model type: %w", err)
	}

	// If the user explicitly set a model type, override the detected type and
	// PreSigmoided flag but keep the detected NumSpecies and EmbeddingDim.
	if cfg.modelType != nil {
		mt := *cfg.modelType
		detectedConfig.ModelType = mt
		// Derive PreSigmoided from the overridden model type.
		detectedConfig.PreSigmoided = (mt == ModelTypeBSGFinland)
		// Update sample rate / duration / sample count to match the override.
		switch mt {
		case ModelTypeBirdNetV24, ModelTypeBSGFinland:
			detectedConfig.SampleRate = 48000
			detectedConfig.Duration = 3.0
			detectedConfig.SampleCount = SampleCountV24
		case ModelTypeBirdNetV30:
			detectedConfig.SampleRate = 32000
			detectedConfig.Duration = 5.0
			detectedConfig.SampleCount = SampleCountV30
		case ModelTypePerchV2:
			detectedConfig.SampleRate = 32000
			detectedConfig.Duration = 5.0
			detectedConfig.SampleCount = SampleCountPerch
		}
	}

	// Validate label count against model output dimension.
	if len(labels) != detectedConfig.NumSpecies {
		return nil, fmt.Errorf("%w: got %d labels, model expects %d",
			ErrLabelCount, len(labels), detectedConfig.NumSpecies)
	}

	// Get input/output info from the model file for tensor names and shapes.
	ortInputs, ortOutputs, err := ort.GetInputOutputInfo(cfg.modelPath)
	if err != nil {
		return nil, fmt.Errorf("birdnet: reading model info: %w", err)
	}

	inputNames := make([]string, len(ortInputs))
	for i, info := range ortInputs {
		inputNames[i] = info.Name
	}
	outputNames := make([]string, len(ortOutputs))
	for i, info := range ortOutputs {
		outputNames[i] = info.Name
	}

	// Check dynamic batch support from input shapes.
	inputInfos := make([]tensorInfo, len(ortInputs))
	for i, info := range ortInputs {
		inputInfos[i] = tensorInfo{
			Name:       info.Name,
			Dimensions: []int64(info.Dimensions),
		}
	}
	dynBatch := dynamicBatchSupported(inputInfos)

	// Create or reuse session options.
	sessOpts := cfg.sessionOpts
	ownsSessOpts := false
	if sessOpts == nil {
		sessOpts, err = ort.NewSessionOptions()
		if err != nil {
			return nil, fmt.Errorf("birdnet: creating session options: %w", err)
		}
		ownsSessOpts = true
	}

	// Apply execution providers.
	for _, p := range cfg.providers {
		if err := p.setup(sessOpts); err != nil {
			if ownsSessOpts {
				sessOpts.Destroy()
			}
			return nil, fmt.Errorf("birdnet: setting up %s provider: %w", p.name, err)
		}
	}

	// Create pre-allocated input tensor: shape [1, sampleCount].
	inputTensor, err := ort.NewEmptyTensor[float32](
		ort.NewShape(1, int64(detectedConfig.SampleCount)),
	)
	if err != nil {
		if ownsSessOpts {
			sessOpts.Destroy()
		}
		return nil, fmt.Errorf("birdnet: creating input tensor: %w", err)
	}

	// Create pre-allocated output tensors based on model type.
	outputTensors, err := createOutputTensors(detectedConfig, ortOutputs)
	if err != nil {
		inputTensor.Destroy()
		if ownsSessOpts {
			sessOpts.Destroy()
		}
		return nil, fmt.Errorf("birdnet: creating output tensors: %w", err)
	}

	// Build Value slices for the AdvancedSession constructor.
	inputs := []ort.Value{inputTensor}
	outputs := make([]ort.Value, len(outputTensors))
	for i, t := range outputTensors {
		outputs[i] = t
	}

	session, err := ort.NewAdvancedSession(
		cfg.modelPath, inputNames, outputNames, inputs, outputs, sessOpts,
	)
	if err != nil {
		inputTensor.Destroy()
		for _, t := range outputTensors {
			t.Destroy()
		}
		if ownsSessOpts {
			sessOpts.Destroy()
		}
		return nil, fmt.Errorf("birdnet: creating ONNX session: %w", err)
	}

	return &Classifier{
		session:       session,
		config:        detectedConfig,
		labels:        labels,
		topK:          cfg.topK,
		minConf:       cfg.minConf,
		dynamicBatch:  dynBatch,
		inputTensor:   inputTensor,
		outputTensors: outputTensors,
		sessionOpts:   sessOpts,
		modelPath:     cfg.modelPath,
		inputNames:    inputNames,
		outputNames:   outputNames,
	}, nil
}

// createOutputTensors allocates the output tensors for a given model type.
func createOutputTensors(cfg ModelConfig, ortOutputs []ort.InputOutputInfo) ([]*ort.Tensor[float32], error) {
	switch cfg.ModelType {
	case ModelTypeBirdNetV24, ModelTypeBSGFinland:
		// 1 output: [1, numSpecies]
		t, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(cfg.NumSpecies)))
		if err != nil {
			return nil, err
		}
		return []*ort.Tensor[float32]{t}, nil

	case ModelTypeBirdNetV30:
		// 2 outputs: [1, embeddingDim] and [1, numSpecies]
		tEmbed, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(cfg.EmbeddingDim)))
		if err != nil {
			return nil, err
		}
		tSpecies, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(cfg.NumSpecies)))
		if err != nil {
			tEmbed.Destroy()
			return nil, err
		}
		return []*ort.Tensor[float32]{tEmbed, tSpecies}, nil

	case ModelTypePerchV2:
		// 4 outputs: use shapes from the ONNX model info.
		if len(ortOutputs) < 4 {
			return nil, fmt.Errorf("birdnet: Perch model requires 4 outputs, got %d", len(ortOutputs))
		}
		tensors := make([]*ort.Tensor[float32], 4)
		for i := range 4 {
			dims := ortOutputs[i].Dimensions
			shape := make([]int64, len(dims))
			for j, d := range dims {
				if d <= 0 {
					// Replace dynamic dimensions with 1 for single-segment inference.
					shape[j] = 1
				} else {
					shape[j] = d
				}
			}
			t, err := ort.NewEmptyTensor[float32](ort.NewShape(shape...))
			if err != nil {
				for k := range i {
					tensors[k].Destroy()
				}
				return nil, err
			}
			tensors[i] = t
		}
		return tensors, nil

	default:
		return nil, fmt.Errorf("birdnet: unsupported model type %v", cfg.ModelType)
	}
}

// Predict runs inference on a single audio segment and returns the prediction result.
// The segment must have exactly config.SampleCount float32 samples.
func (c *Classifier) Predict(ctx context.Context, segment []float32) (*PredictionResult, error) {
	if len(segment) != c.config.SampleCount {
		return nil, &InputSizeError{
			Expected: c.config.SampleCount,
			Got:      len(segment),
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Copy segment data into the pre-allocated input tensor.
	copy(c.inputTensor.GetData(), segment)

	// Run inference, respecting context cancellation if applicable.
	if err := c.runSession(ctx); err != nil {
		return nil, fmt.Errorf("birdnet: running inference: %w", err)
	}

	return c.buildResult(), nil
}

// runSession runs the ONNX session, optionally with context cancellation support.
// The caller must hold c.mu.
func (c *Classifier) runSession(ctx context.Context) error {
	// Fast path: context.Background() or context.TODO() with no deadline/cancel.
	_, hasDeadline := ctx.Deadline()
	if !hasDeadline && ctx.Done() == nil {
		return c.session.Run()
	}

	// Cancellable path: use RunOptions with a goroutine watching ctx.Done().
	runOpts, err := ort.NewRunOptions()
	if err != nil {
		return fmt.Errorf("creating run options: %w", err)
	}

	// Watch for cancellation in a separate goroutine.
	// The goroutine signals exited when it is completely done, so we can
	// safely destroy runOpts only after the goroutine has exited.
	done := make(chan struct{})
	exited := make(chan struct{})
	go func() {
		defer close(exited)
		select {
		case <-ctx.Done():
			runOpts.Terminate()
		case <-done:
		}
	}()

	runErr := c.session.RunWithOptions(runOpts)

	// Signal the watcher goroutine to stop, then wait for it to exit
	// before destroying runOpts to avoid a Terminate/Destroy race.
	close(done)
	<-exited

	runOpts.Destroy()

	return runErr
}

// buildResult constructs a PredictionResult from the current output tensor data.
// The caller must hold c.mu.
func (c *Classifier) buildResult() *PredictionResult {
	var logits []float32
	var embeddings []float32

	switch c.config.ModelType {
	case ModelTypeBirdNetV24, ModelTypeBSGFinland:
		logits = c.outputTensors[0].GetData()
	case ModelTypeBirdNetV30:
		embeddings = c.outputTensors[0].GetData()
		logits = c.outputTensors[1].GetData()
	case ModelTypePerchV2:
		embeddings = c.outputTensors[0].GetData()
		logits = c.outputTensors[1].GetData()
	}

	// Get top-K predictions.
	predictions := TopKPredictions(logits, c.labels, c.topK, c.minConf, c.config.PreSigmoided)

	// Compute raw scores (all species after activation).
	rawScores := make([]float32, len(logits))
	if c.config.PreSigmoided {
		copy(rawScores, logits)
	} else {
		for i, v := range logits {
			rawScores[i] = Sigmoid(v)
		}
	}

	// Copy embeddings if present (don't retain a reference to tensor buffer).
	var embeddingsCopy []float32
	if len(embeddings) > 0 {
		embeddingsCopy = make([]float32, len(embeddings))
		copy(embeddingsCopy, embeddings)
	}

	return &PredictionResult{
		ModelType:   c.config.ModelType,
		Predictions: predictions,
		Embeddings:  embeddingsCopy,
		RawScores:   rawScores,
	}
}

// PredictBatch runs inference on multiple audio segments.
// If the model supports dynamic batching, segments are processed in a single
// inference pass. Otherwise, they are processed sequentially.
func (c *Classifier) PredictBatch(ctx context.Context, segments [][]float32) ([]*PredictionResult, error) {
	// Validate all segments first.
	for i, seg := range segments {
		if len(seg) != c.config.SampleCount {
			return nil, fmt.Errorf("%w: segment %d has %d samples, expected %d",
				ErrBatchInputSize, i, len(seg), c.config.SampleCount)
		}
	}

	if len(segments) == 0 {
		return []*PredictionResult{}, nil
	}

	// If dynamic batch is not supported, fall back to sequential processing.
	if !c.dynamicBatch {
		return c.predictSequential(ctx, segments)
	}

	return c.predictDynamicBatch(ctx, segments)
}

// predictSequential processes segments one at a time using Predict.
func (c *Classifier) predictSequential(ctx context.Context, segments [][]float32) ([]*PredictionResult, error) {
	results := make([]*PredictionResult, len(segments))
	for i, seg := range segments {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		result, err := c.Predict(ctx, seg)
		if err != nil {
			return nil, fmt.Errorf("birdnet: batch segment %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// predictDynamicBatch processes all segments in a single inference pass using
// a DynamicAdvancedSession.
func (c *Classifier) predictDynamicBatch(ctx context.Context, segments [][]float32) ([]*PredictionResult, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	batchSize := len(segments)
	sampleCount := c.config.SampleCount

	// Lazily create the batch session.
	if c.batchSession == nil {
		var err error
		c.batchSession, err = ort.NewDynamicAdvancedSession(
			c.modelPath, c.inputNames, c.outputNames, c.sessionOpts,
		)
		if err != nil {
			return nil, fmt.Errorf("birdnet: creating batch session: %w", err)
		}
	}

	// Create batch input tensor: [batchSize, sampleCount].
	flatInput := make([]float32, batchSize*sampleCount)
	for i, seg := range segments {
		copy(flatInput[i*sampleCount:], seg)
	}
	inputTensor, err := ort.NewTensor(
		ort.NewShape(int64(batchSize), int64(sampleCount)), flatInput,
	)
	if err != nil {
		return nil, fmt.Errorf("birdnet: creating batch input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create batch output tensors.
	outputTensors, err := c.createBatchOutputTensors(batchSize)
	if err != nil {
		return nil, fmt.Errorf("birdnet: creating batch output tensors: %w", err)
	}
	defer func() {
		for _, t := range outputTensors {
			t.Destroy()
		}
	}()

	// Build Value slices.
	inputs := []ort.Value{inputTensor}
	outputs := make([]ort.Value, len(outputTensors))
	for i, t := range outputTensors {
		outputs[i] = t
	}

	// Check for cancellation before running inference.
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Run inference.
	if err := c.batchSession.Run(inputs, outputs); err != nil {
		return nil, fmt.Errorf("birdnet: running batch inference: %w", err)
	}

	// Split results per segment.
	return c.splitBatchResults(outputTensors, batchSize), nil
}

// createBatchOutputTensors creates output tensors sized for a batch.
func (c *Classifier) createBatchOutputTensors(batchSize int) ([]*ort.Tensor[float32], error) {
	bs := int64(batchSize)
	switch c.config.ModelType {
	case ModelTypeBirdNetV24, ModelTypeBSGFinland:
		t, err := ort.NewEmptyTensor[float32](ort.NewShape(bs, int64(c.config.NumSpecies)))
		if err != nil {
			return nil, err
		}
		return []*ort.Tensor[float32]{t}, nil

	case ModelTypeBirdNetV30:
		tEmbed, err := ort.NewEmptyTensor[float32](ort.NewShape(bs, int64(c.config.EmbeddingDim)))
		if err != nil {
			return nil, err
		}
		tSpecies, err := ort.NewEmptyTensor[float32](ort.NewShape(bs, int64(c.config.NumSpecies)))
		if err != nil {
			tEmbed.Destroy()
			return nil, err
		}
		return []*ort.Tensor[float32]{tEmbed, tSpecies}, nil

	case ModelTypePerchV2:
		// For Perch batch, use the single-segment output shapes but replace batch dim.
		tensors := make([]*ort.Tensor[float32], len(c.outputTensors))
		for i, st := range c.outputTensors {
			origShape := st.GetShape()
			shape := make([]int64, len(origShape))
			copy(shape, []int64(origShape))
			if len(shape) > 0 {
				shape[0] = bs
			}
			t, err := ort.NewEmptyTensor[float32](ort.NewShape(shape...))
			if err != nil {
				for k := range i {
					tensors[k].Destroy()
				}
				return nil, err
			}
			tensors[i] = t
		}
		return tensors, nil

	default:
		return nil, fmt.Errorf("birdnet: unsupported model type for batch: %v", c.config.ModelType)
	}
}

// splitBatchResults splits batch output tensors into per-segment PredictionResults.
func (c *Classifier) splitBatchResults(outputTensors []*ort.Tensor[float32], batchSize int) []*PredictionResult {
	results := make([]*PredictionResult, batchSize)
	numSpecies := c.config.NumSpecies
	embeddingDim := c.config.EmbeddingDim

	for i := range batchSize {
		var logits []float32
		var embeddings []float32

		switch c.config.ModelType {
		case ModelTypeBirdNetV24, ModelTypeBSGFinland:
			allLogits := outputTensors[0].GetData()
			logits = allLogits[i*numSpecies : (i+1)*numSpecies]
		case ModelTypeBirdNetV30:
			allEmbed := outputTensors[0].GetData()
			allLogits := outputTensors[1].GetData()
			embeddings = allEmbed[i*embeddingDim : (i+1)*embeddingDim]
			logits = allLogits[i*numSpecies : (i+1)*numSpecies]
		case ModelTypePerchV2:
			allEmbed := outputTensors[0].GetData()
			allLogits := outputTensors[1].GetData()
			embeddings = allEmbed[i*embeddingDim : (i+1)*embeddingDim]
			logits = allLogits[i*numSpecies : (i+1)*numSpecies]
		}

		predictions := TopKPredictions(logits, c.labels, c.topK, c.minConf, c.config.PreSigmoided)

		rawScores := make([]float32, len(logits))
		if c.config.PreSigmoided {
			copy(rawScores, logits)
		} else {
			for j, v := range logits {
				rawScores[j] = Sigmoid(v)
			}
		}

		var embeddingsCopy []float32
		if len(embeddings) > 0 {
			embeddingsCopy = make([]float32, len(embeddings))
			copy(embeddingsCopy, embeddings)
		}

		results[i] = &PredictionResult{
			ModelType:   c.config.ModelType,
			Predictions: predictions,
			Embeddings:  embeddingsCopy,
			RawScores:   rawScores,
		}
	}

	return results
}

// Close releases all ONNX Runtime resources held by the Classifier.
// It is safe to call Close multiple times.
func (c *Classifier) Close() error {
	c.closeOnce.Do(func() {
		if c.session != nil {
			c.session.Destroy()
		}
		for _, t := range c.outputTensors {
			t.Destroy()
		}
		if c.inputTensor != nil {
			c.inputTensor.Destroy()
		}
		if c.batchSession != nil {
			c.batchSession.Destroy()
		}
		if c.sessionOpts != nil {
			c.sessionOpts.Destroy()
		}
	})
	return nil
}

// Config returns the detected model configuration.
func (c *Classifier) Config() ModelConfig {
	return c.config
}

// Labels returns a copy of the species labels used by the Classifier.
func (c *Classifier) Labels() []string {
	out := make([]string, len(c.labels))
	copy(out, c.labels)
	return out
}
