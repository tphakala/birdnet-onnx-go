package birdnet

import (
	"container/heap"
	"math"
	"sort"
)

// Sigmoid computes the sigmoid activation function: 1 / (1 + exp(-x)).
func Sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

// scoreEntry holds a score index for the min-heap.
type scoreEntry struct {
	index int
	score float32
}

// minHeap implements heap.Interface for scoreEntry, ordered by score ascending.
type minHeap []scoreEntry

func (h *minHeap) Len() int            { return len(*h) }
func (h *minHeap) Less(i, j int) bool  { return (*h)[i].score < (*h)[j].score }
func (h *minHeap) Swap(i, j int)       { (*h)[i], (*h)[j] = (*h)[j], (*h)[i] }

func (h *minHeap) Push(x any) {
	entry, ok := x.(scoreEntry)
	if !ok {
		return
	}
	*h = append(*h, entry)
}

func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// TopKPredictions selects the top-k highest-scoring predictions from raw model
// output using a min-heap for O(n log k) selection. Sigmoid is applied only to
// the selected k scores (not all n). If preSigmoided is true (BSG model),
// sigmoid is skipped. Results below minConf are filtered out after activation.
func TopKPredictions(scores []float32, labels []string, k int, minConf float32, preSigmoided bool) []Prediction {
	n := len(scores)
	if n == 0 || k <= 0 || len(labels) == 0 {
		return []Prediction{}
	}
	if k > n {
		k = n
	}
	// Clamp to label count to avoid out-of-bounds access.
	if len(labels) < n {
		n = len(labels)
	}

	h := topKHeap(scores, n, k)

	results := collectPredictions(h, labels, minConf, preSigmoided)

	// Sort descending by confidence.
	sort.Slice(results, func(i, j int) bool {
		return results[i].Confidence > results[j].Confidence
	})

	return results
}

// topKHeap builds a min-heap of the top-k raw scores from scores[:n].
func topKHeap(scores []float32, n, k int) *minHeap {
	h := &minHeap{}
	heap.Init(h)

	for i := range n {
		s := scores[i]
		if h.Len() < k {
			heap.Push(h, scoreEntry{index: i, score: s})
		} else if s > (*h)[0].score {
			(*h)[0] = scoreEntry{index: i, score: s}
			heap.Fix(h, 0)
		}
	}

	return h
}

// collectPredictions drains the heap, applies activation, and filters by minConf.
func collectPredictions(h *minHeap, labels []string, minConf float32, preSigmoided bool) []Prediction {
	results := make([]Prediction, 0, h.Len())
	for h.Len() > 0 {
		popped := heap.Pop(h)
		entry, ok := popped.(scoreEntry)
		if !ok {
			continue
		}
		conf := entry.score
		if !preSigmoided {
			conf = Sigmoid(entry.score)
		}
		if conf >= minConf {
			results = append(results, Prediction{
				Species:    labels[entry.index],
				Confidence: conf,
				Index:      entry.index,
			})
		}
	}
	return results
}
