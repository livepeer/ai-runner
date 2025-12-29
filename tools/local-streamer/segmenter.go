package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/livepeer/go-livepeer/trickle"
)

// Segmenter handles FFmpeg segmentation and Trickle publishing
type Segmenter struct {
	inputFile  string
	publishURL string
	segmentDur float64 // segment duration in seconds
	realtime   bool    // whether to stream in realtime (-re flag)
}

// NewSegmenter creates a new segmenter
func NewSegmenter(inputFile, publishURL string, segmentDur float64, realtime bool) *Segmenter {
	return &Segmenter{
		inputFile:  inputFile,
		publishURL: publishURL,
		segmentDur: segmentDur,
		realtime:   realtime,
	}
}

// Start begins segmentation and publishing
func (s *Segmenter) Start(ctx context.Context) error {
	slog.Info("Starting segmenter", "input", s.inputFile, "publishURL", s.publishURL, "segmentDur", s.segmentDur)

	// Create trickle publisher
	publisher, err := trickle.NewTricklePublisher(s.publishURL)
	if err != nil {
		return fmt.Errorf("failed to create trickle publisher: %w", err)
	}

	// Create a temporary directory for segment pipes
	tmpDir, err := os.MkdirTemp("", "local-streamer-*")
	if err != nil {
		return fmt.Errorf("failed to create temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create the segment pattern for named pipes
	segmentPattern := filepath.Join(tmpDir, "segment-%d.ts")

	// Create first named pipe
	firstPipe := fmt.Sprintf(segmentPattern, 0)
	if err := syscall.Mkfifo(firstPipe, 0666); err != nil && !os.IsExist(err) {
		return fmt.Errorf("failed to create first pipe: %w", err)
	}

	// Channel to signal completion
	doneCh := make(chan struct{})
	var wg sync.WaitGroup

	// Start the segment processor in a goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		s.processSegments(ctx, segmentPattern, publisher, doneCh)
	}()

	// Build FFmpeg command
	args := []string{}
	if s.realtime {
		args = append(args, "-re") // Read input at native frame rate
	}
	args = append(args,
		"-i", s.inputFile,
		"-c:v", "libx264",
		"-preset", "ultrafast",
		"-tune", "zerolatency",
		"-c:a", "aac",
		"-f", "segment",
		"-segment_time", fmt.Sprintf("%.2f", s.segmentDur),
		"-reset_timestamps", "1",
		segmentPattern,
	)

	cmd := exec.CommandContext(ctx, "ffmpeg", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Set up graceful termination
	cmd.Cancel = func() error {
		return cmd.Process.Signal(syscall.SIGTERM)
	}
	cmd.WaitDelay = 5 * time.Second

	slog.Info("Running FFmpeg", "args", args)
	err = cmd.Run()

	// Signal completion to the segment processor
	close(doneCh)

	// Wait for segment processor to finish
	wg.Wait()

	// Close the publisher
	if closeErr := publisher.Close(); closeErr != nil {
		slog.Error("Error closing publisher", "err", closeErr)
	}

	if err != nil && ctx.Err() == nil {
		return fmt.Errorf("ffmpeg error: %w", err)
	}

	slog.Info("Segmenter finished")
	return nil
}

// processSegments reads segments from named pipes and publishes them
func (s *Segmenter) processSegments(ctx context.Context, pattern string, publisher *trickle.TricklePublisher, doneCh <-chan struct{}) {
	segmentNum := 0
	for {
		select {
		case <-ctx.Done():
			slog.Info("Segment processor cancelled")
			return
		case <-doneCh:
			slog.Info("Segment processor done signal received")
			return
		default:
		}

		pipeName := fmt.Sprintf(pattern, segmentNum)
		nextPipeName := fmt.Sprintf(pattern, segmentNum+1)

		// Create next pipe ahead of time
		if err := syscall.Mkfifo(nextPipeName, 0666); err != nil && !os.IsExist(err) {
			slog.Error("Failed to create next pipe", "pipe", nextPipeName, "err", err)
		}

		// Open pipe for reading with timeout
		file, err := openPipeWithTimeout(pipeName, 20*time.Second, doneCh)
		if err != nil {
			slog.Info("Pipe open completed or timed out", "pipe", pipeName, "err", err)
			// Clean up pipes
			os.Remove(pipeName)
			os.Remove(nextPipeName)
			return
		}

		// Read and publish the segment
		if err := s.publishSegment(ctx, file, publisher, segmentNum); err != nil {
			slog.Error("Failed to publish segment", "segment", segmentNum, "err", err)
		}

		file.Close()
		os.Remove(pipeName)
		segmentNum++
	}
}

// openPipeWithTimeout opens a named pipe with a timeout
func openPipeWithTimeout(name string, timeout time.Duration, doneCh <-chan struct{}) (*os.File, error) {
	resultCh := make(chan *os.File, 1)
	errCh := make(chan error, 1)

	go func() {
		// Open in non-blocking mode first
		file, err := os.OpenFile(name, os.O_RDONLY, 0)
		if err != nil {
			errCh <- err
			return
		}
		resultCh <- file
	}()

	select {
	case file := <-resultCh:
		return file, nil
	case err := <-errCh:
		return nil, err
	case <-doneCh:
		return nil, fmt.Errorf("done signal received")
	case <-time.After(timeout):
		return nil, fmt.Errorf("timeout waiting for pipe")
	}
}

// publishSegment reads from a file and publishes to trickle
func (s *Segmenter) publishSegment(ctx context.Context, file *os.File, publisher *trickle.TricklePublisher, segmentNum int) error {
	startTime := time.Now()
	reader := bufio.NewReader(file)

	// Read all data into a buffer first to get the size
	data, err := io.ReadAll(reader)
	if err != nil {
		return fmt.Errorf("failed to read segment: %w", err)
	}

	if len(data) == 0 {
		slog.Warn("Empty segment, skipping", "segment", segmentNum)
		return nil
	}

	// Create a reader from the data
	dataReader := &byteReader{data: data}

	// Publish the segment
	if err := publisher.Write(dataReader); err != nil {
		return fmt.Errorf("failed to publish segment: %w", err)
	}

	slog.Info("Published segment", "segment", segmentNum, "bytes", len(data), "took", time.Since(startTime))
	return nil
}

// byteReader implements io.Reader for a byte slice
type byteReader struct {
	data []byte
	pos  int
}

func (r *byteReader) Read(p []byte) (n int, err error) {
	if r.pos >= len(r.data) {
		return 0, io.EOF
	}
	n = copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}

