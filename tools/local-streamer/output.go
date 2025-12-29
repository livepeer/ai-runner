package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"time"

	"github.com/livepeer/go-livepeer/trickle"
)

// OutputHandler handles subscribing to trickle output and writing to file
type OutputHandler struct {
	subscribeURL string
	outputFile   string
}

// NewOutputHandler creates a new output handler
func NewOutputHandler(subscribeURL, outputFile string) *OutputHandler {
	return &OutputHandler{
		subscribeURL: subscribeURL,
		outputFile:   outputFile,
	}
}

// Start begins subscribing and writing output
func (o *OutputHandler) Start(ctx context.Context) error {
	slog.Info("Starting output handler", "subscribeURL", o.subscribeURL, "outputFile", o.outputFile)

	// Create the output file
	file, err := os.Create(o.outputFile)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer file.Close()

	// Create trickle subscriber
	subscriber, err := trickle.NewTrickleSubscriber(trickle.TrickleSubscriberConfig{
		URL: o.subscribeURL,
		Ctx: ctx,
	})
	if err != nil {
		return fmt.Errorf("failed to create trickle subscriber: %w", err)
	}

	var totalBytes int64
	segmentCount := 0
	retries := 0
	const maxRetries = 60 // Wait up to 30 seconds for first segment
	const retryPause = 500 * time.Millisecond

	for {
		select {
		case <-ctx.Done():
			slog.Info("Output handler cancelled", "totalBytes", totalBytes, "segments", segmentCount)
			return nil
		default:
		}

		// Read from trickle
		segment, err := subscriber.Read()
		if err != nil {
			if errors.Is(err, trickle.EOS) {
				slog.Info("End of stream reached", "totalBytes", totalBytes, "segments", segmentCount)
				return nil
			}
			if errors.Is(err, trickle.StreamNotFoundErr) {
				// Stream might not exist yet, retry
				if retries < maxRetries {
					retries++
					slog.Debug("Stream not found, retrying", "retry", retries)
					time.Sleep(retryPause)
					continue
				}
				return fmt.Errorf("stream not found after %d retries", maxRetries)
			}

			// Handle sequence nonexistent error
			var seqErr *trickle.SequenceNonexistent
			if errors.As(err, &seqErr) {
				// Skip to leading edge
				slog.Info("Sequence not found, skipping to latest", "requested", seqErr.Seq, "latest", seqErr.Latest)
				subscriber.SetSeq(seqErr.Latest)
				continue
			}

			// Other errors - retry with backoff
			if retries < maxRetries {
				retries++
				slog.Warn("Error reading segment, retrying", "err", err, "retry", retries)
				time.Sleep(retryPause)
				continue
			}
			return fmt.Errorf("failed to read segment after %d retries: %w", maxRetries, err)
		}

		// Reset retry counter on successful read
		retries = 0

		// Get segment sequence number
		seq := trickle.GetSeq(segment)

		// Copy segment data to file
		startTime := time.Now()
		n, err := io.Copy(file, segment.Body)
		segment.Body.Close()

		if err != nil {
			slog.Error("Error writing segment to file", "seq", seq, "err", err)
			continue
		}

		totalBytes += n
		segmentCount++
		slog.Info("Wrote segment to file", "seq", seq, "bytes", n, "totalBytes", totalBytes, "took", time.Since(startTime))
	}
}

