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
	dynamicBatch  bool
	ownsSessOpts  bool
	mu            sync.Mutex
	closeOnce     sync.Once

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

	if cfg.modelPath == "" {
		return nil, ErrModelPath
	}

	labels, err := resolveLabels(&cfg)
	if err != nil {
		return nil, err
	}

	ortInputs, ortOutputs, err := ort.GetInputOutputInfo(cfg.modelPath)
	if err != nil {
		return nil, fmt.Errorf("birdnet: reading model info: %w", err)
	}

	detectedConfig, err := detectModelTypeFromInfo(ortInputs, ortOutputs)
	if err != nil {
		return nil, fmt.Errorf("birdnet: detecting model type: %w", err)
	}

	applyModelTypeOverride(&detectedConfig, cfg.modelType)

	if len(labels) != detectedConfig.NumSpecies {
		return nil, fmt.Errorf("%w: got %d labels, model expects %d",
			ErrLabelCount, len(labels), detectedConfig.NumSpecies)
	}

	return buildClassifier(&cfg, &detectedConfig, labels, ortInputs, ortOutputs)
}

// resolveLabels loads labels from whichever source was configured.
func resolveLabels(cfg *classifierConfig) ([]string, error) {
	switch {
	case len(cfg.labels) > 0:
		return cfg.labels, nil
	case cfg.labelsPath != "":
		labels, err := LoadLabels(cfg.labelsPath)
		if err != nil {
			return nil, fmt.Errorf("birdnet: loading labels from file: %w", err)
		}
		return labels, nil
	case cfg.labelsReader != nil:
		labels, err := LoadLabelsFromReader(cfg.labelsReader, cfg.labelsFormat)
		if err != nil {
			return nil, fmt.Errorf("birdnet: loading labels from reader: %w", err)
		}
		return labels, nil
	default:
		return nil, ErrLabelsRequired
	}
}

// applyModelTypeOverride applies a user-specified model type override to the detected config.
func applyModelTypeOverride(cfg *ModelConfig, mt *ModelType) {
	if mt == nil {
		return
	}
	overrideType := *mt
	cfg.ModelType = overrideType
	cfg.PreSigmoided = (overrideType == ModelTypeBSGFinland)

	switch overrideType {
	case ModelTypeBirdNetV24, ModelTypeBSGFinland:
		cfg.SampleRate = sampleRate48k
		cfg.Duration = duration3s
		cfg.SampleCount = SampleCountV24
	case ModelTypeBirdNetV30:
		cfg.SampleRate = sampleRate32k
		cfg.Duration = duration5s
		cfg.SampleCount = SampleCountV30
	case ModelTypePerchV2:
		cfg.SampleRate = sampleRate32k
		cfg.Duration = duration5s
		cfg.SampleCount = SampleCountPerch
	}
}

// buildClassifier creates the ONNX session and assembles the Classifier.
func buildClassifier(
	cfg *classifierConfig,
	modelCfg *ModelConfig,
	labels []string,
	ortInputs, ortOutputs []ort.InputOutputInfo,
) (*Classifier, error) {
	inputNames := make([]string, len(ortInputs))
	for i, info := range ortInputs {
		inputNames[i] = info.Name
	}
	outputNames := make([]string, len(ortOutputs))
	for i, info := range ortOutputs {
		outputNames[i] = info.Name
	}

	inputInfos := make([]tensorInfo, len(ortInputs))
	for i, info := range ortInputs {
		inputInfos[i] = tensorInfo{
			Name:       info.Name,
			Dimensions: []int64(info.Dimensions),
		}
	}
	dynBatch := dynamicBatchSupported(inputInfos)

	sessOpts, ownsSessOpts, err := resolveSessionOpts(cfg)
	if err != nil {
		return nil, err
	}
	cleanup := func() {
		if ownsSessOpts {
			_ = sessOpts.Destroy()
		}
	}

	for _, p := range cfg.providers {
		if err := p.setup(sessOpts); err != nil {
			cleanup()
			return nil, fmt.Errorf("birdnet: setting up %s provider: %w", p.name, err)
		}
	}

	inputTensor, err := ort.NewEmptyTensor[float32](
		ort.NewShape(1, int64(modelCfg.SampleCount)),
	)
	if err != nil {
		cleanup()
		return nil, fmt.Errorf("birdnet: creating input tensor: %w", err)
	}

	outputTensors, err := createOutputTensors(*modelCfg, ortOutputs)
	if err != nil {
		_ = inputTensor.Destroy()
		cleanup()
		return nil, fmt.Errorf("birdnet: creating output tensors: %w", err)
	}

	inputs := []ort.Value{inputTensor}
	outputs := make([]ort.Value, len(outputTensors))
	for i, t := range outputTensors {
		outputs[i] = t
	}

	session, err := ort.NewAdvancedSession(
		cfg.modelPath, inputNames, outputNames, inputs, outputs, sessOpts,
	)
	if err != nil {
		_ = inputTensor.Destroy()
		destroyTensors(outputTensors)
		cleanup()
		return nil, fmt.Errorf("birdnet: creating ONNX session: %w", err)
	}

	return &Classifier{
		session:       session,
		config:        *modelCfg,
		labels:        labels,
		topK:          cfg.topK,
		minConf:       cfg.minConf,
		dynamicBatch:  dynBatch,
		ownsSessOpts:  ownsSessOpts,
		inputTensor:   inputTensor,
		outputTensors: outputTensors,
		sessionOpts:   sessOpts,
		modelPath:     cfg.modelPath,
		inputNames:    inputNames,
		outputNames:   outputNames,
	}, nil
}

// resolveSessionOpts creates or reuses session options.
func resolveSessionOpts(cfg *classifierConfig) (*ort.SessionOptions, bool, error) {
	if cfg.sessionOpts != nil {
		return cfg.sessionOpts, false, nil
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, false, fmt.Errorf("birdnet: creating session options: %w", err)
	}
	return opts, true, nil
}

// destroyTensors releases a slice of tensors, ignoring errors.
func destroyTensors(tensors []*ort.Tensor[float32]) {
	for _, t := range tensors {
		_ = t.Destroy()
	}
}

// createOutputTensors allocates the output tensors for a given model type.
func createOutputTensors(cfg ModelConfig, ortOutputs []ort.InputOutputInfo) ([]*ort.Tensor[float32], error) {
	switch cfg.ModelType {
	case ModelTypeBirdNetV24, ModelTypeBSGFinland:
		return createSingleOutputTensor(cfg.NumSpecies)

	case ModelTypeBirdNetV30:
		return createEmbeddingAndSpeciesTensors(cfg.EmbeddingDim, cfg.NumSpecies)

	case ModelTypePerchV2:
		return createPerchOutputTensors(ortOutputs)

	default:
		return nil, fmt.Errorf("birdnet: unsupported model type %v", cfg.ModelType)
	}
}

func createSingleOutputTensor(numSpecies int) ([]*ort.Tensor[float32], error) {
	t, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(numSpecies)))
	if err != nil {
		return nil, err
	}
	return []*ort.Tensor[float32]{t}, nil
}

func createEmbeddingAndSpeciesTensors(embeddingDim, numSpecies int) ([]*ort.Tensor[float32], error) {
	tEmbed, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(embeddingDim)))
	if err != nil {
		return nil, err
	}
	tSpecies, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(numSpecies)))
	if err != nil {
		_ = tEmbed.Destroy()
		return nil, err
	}
	return []*ort.Tensor[float32]{tEmbed, tSpecies}, nil
}

func createPerchOutputTensors(ortOutputs []ort.InputOutputInfo) ([]*ort.Tensor[float32], error) {
	if len(ortOutputs) < outputCountPerch {
		return nil, fmt.Errorf("birdnet: Perch model requires %d outputs, got %d", outputCountPerch, len(ortOutputs))
	}
	tensors := make([]*ort.Tensor[float32], outputCountPerch)
	for i := range outputCountPerch {
		dims := ortOutputs[i].Dimensions
		shape := make([]int64, len(dims))
		for j, d := range dims {
			if d <= 0 {
				shape[j] = 1
			} else {
				shape[j] = d
			}
		}
		t, err := ort.NewEmptyTensor[float32](ort.NewShape(shape...))
		if err != nil {
			destroyTensors(tensors[:i])
			return nil, err
		}
		tensors[i] = t
	}
	return tensors, nil
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

	copy(c.inputTensor.GetData(), segment)

	if err := c.runSession(ctx); err != nil {
		return nil, fmt.Errorf("birdnet: running inference: %w", err)
	}

	return c.buildResult(), nil
}

// runSession runs the ONNX session, optionally with context cancellation support.
// The caller must hold c.mu.
func (c *Classifier) runSession(ctx context.Context) error {
	_, hasDeadline := ctx.Deadline()
	if !hasDeadline && ctx.Done() == nil {
		return c.session.Run()
	}

	runOpts, err := ort.NewRunOptions()
	if err != nil {
		return fmt.Errorf("creating run options: %w", err)
	}

	done := make(chan struct{})
	exited := make(chan struct{})
	go func() {
		defer close(exited)
		select {
		case <-ctx.Done():
			_ = runOpts.Terminate()
		case <-done:
		}
	}()

	runErr := c.session.RunWithOptions(runOpts)

	close(done)
	<-exited

	_ = runOpts.Destroy()

	return runErr
}

// buildResult constructs a PredictionResult from the current output tensor data.
// The caller must hold c.mu.
func (c *Classifier) buildResult() *PredictionResult {
	logits, embeddings := c.extractOutputs()
	return c.assemblePredictionResult(logits, embeddings)
}

// extractOutputs reads logit and embedding slices from the output tensors.
func (c *Classifier) extractOutputs() (logits, embeddings []float32) {
	switch c.config.ModelType {
	case ModelTypeBirdNetV24, ModelTypeBSGFinland:
		logits = c.outputTensors[0].GetData()
	case ModelTypeBirdNetV30, ModelTypePerchV2:
		embeddings = c.outputTensors[0].GetData()
		logits = c.outputTensors[1].GetData()
	}
	return logits, embeddings
}

// assemblePredictionResult builds a PredictionResult from raw logits and embeddings.
func (c *Classifier) assemblePredictionResult(logits, embeddings []float32) *PredictionResult {
	predictions := TopKPredictions(logits, c.labels, c.topK, c.minConf, c.config.PreSigmoided)

	rawScores := make([]float32, len(logits))
	if c.config.PreSigmoided {
		copy(rawScores, logits)
	} else {
		for i, v := range logits {
			rawScores[i] = Sigmoid(v)
		}
	}

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
	for i, seg := range segments {
		if len(seg) != c.config.SampleCount {
			return nil, fmt.Errorf("%w: segment %d has %d samples, expected %d",
				ErrBatchInputSize, i, len(seg), c.config.SampleCount)
		}
	}

	if len(segments) == 0 {
		return []*PredictionResult{}, nil
	}

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

	if c.batchSession == nil {
		var err error
		c.batchSession, err = ort.NewDynamicAdvancedSession(
			c.modelPath, c.inputNames, c.outputNames, c.sessionOpts,
		)
		if err != nil {
			return nil, fmt.Errorf("birdnet: creating batch session: %w", err)
		}
	}

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
	defer func() { _ = inputTensor.Destroy() }()

	outputTensors, err := c.createBatchOutputTensors(batchSize)
	if err != nil {
		return nil, fmt.Errorf("birdnet: creating batch output tensors: %w", err)
	}
	defer func() { destroyTensors(outputTensors) }()

	inputs := []ort.Value{inputTensor}
	outputs := make([]ort.Value, len(outputTensors))
	for i, t := range outputTensors {
		outputs[i] = t
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if err := c.batchSession.Run(inputs, outputs); err != nil {
		return nil, fmt.Errorf("birdnet: running batch inference: %w", err)
	}

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
			_ = tEmbed.Destroy()
			return nil, err
		}
		return []*ort.Tensor[float32]{tEmbed, tSpecies}, nil

	case ModelTypePerchV2:
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
				destroyTensors(tensors[:i])
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
		var logits, embeddings []float32

		switch c.config.ModelType {
		case ModelTypeBirdNetV24, ModelTypeBSGFinland:
			allLogits := outputTensors[0].GetData()
			logits = allLogits[i*numSpecies : (i+1)*numSpecies]
		case ModelTypeBirdNetV30, ModelTypePerchV2:
			allEmbed := outputTensors[0].GetData()
			allLogits := outputTensors[1].GetData()
			embeddings = allEmbed[i*embeddingDim : (i+1)*embeddingDim]
			logits = allLogits[i*numSpecies : (i+1)*numSpecies]
		}

		results[i] = c.assemblePredictionResult(logits, embeddings)
	}

	return results
}

// Close releases all ONNX Runtime resources held by the Classifier.
// It is safe to call Close multiple times.
func (c *Classifier) Close() error {
	c.closeOnce.Do(func() {
		if c.session != nil {
			_ = c.session.Destroy()
		}
		destroyTensors(c.outputTensors)
		if c.inputTensor != nil {
			_ = c.inputTensor.Destroy()
		}
		if c.batchSession != nil {
			_ = c.batchSession.Destroy()
		}
		if c.sessionOpts != nil && c.ownsSessOpts {
			_ = c.sessionOpts.Destroy()
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
