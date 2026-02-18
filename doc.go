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
