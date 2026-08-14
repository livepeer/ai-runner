package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

// Config holds the CLI configuration
type Config struct {
	InputFile   string
	OutputFile  string
	RunnerURL   string
	Port        int
	Params      string
	SegmentDur  float64
	Realtime    bool
	NoTranscode bool
}

// StartStreamRequest is the request body for starting a stream
type StartStreamRequest struct {
	SubscribeURL string                 `json:"subscribe_url"`
	PublishURL   string                 `json:"publish_url"`
	ControlURL   string                 `json:"control_url"`
	EventsURL    string                 `json:"events_url"`
	Params       map[string]interface{} `json:"params"`
}

func main() {
	config := parseFlags()

	// Set up logging
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	})))

	slog.Info("Starting local streamer",
		"input", config.InputFile,
		"output", config.OutputFile,
		"runner", config.RunnerURL,
		"port", config.Port,
	)

	// Set up context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle shutdown signals
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		slog.Info("Received signal, shutting down", "signal", sig)
		cancel()
	}()

	// Run the streamer
	if err := run(ctx, config); err != nil {
		slog.Error("Streamer failed", "err", err)
		os.Exit(1)
	}

	slog.Info("Local streamer finished")
}

func parseFlags() Config {
	config := Config{}

	flag.StringVar(&config.InputFile, "input", "", "Input video file path (required)")
	flag.StringVar(&config.OutputFile, "output", "output.ts", "Output file path")
	flag.StringVar(&config.RunnerURL, "runner-url", "http://localhost:8000", "AI Runner URL")
	flag.IntVar(&config.Port, "port", 9935, "Trickle server port")
	flag.StringVar(&config.Params, "params", "{}", "Pipeline parameters as JSON")
	flag.Float64Var(&config.SegmentDur, "segment-dur", 0.5, "Segment duration in seconds")
	flag.BoolVar(&config.Realtime, "realtime", true, "Stream at realtime speed (use -realtime=false to stream as fast as possible)")
	flag.BoolVar(&config.NoTranscode, "no-transcode", false, "Skip transcoding, just copy streams (input must be MPEG-TS compatible)")

	flag.Parse()

	if config.InputFile == "" {
		fmt.Fprintln(os.Stderr, "Error: --input is required")
		flag.Usage()
		os.Exit(1)
	}

	// Check if input file exists
	if _, err := os.Stat(config.InputFile); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Error: input file does not exist: %s\n", config.InputFile)
		os.Exit(1)
	}

	return config
}

func run(parentCtx context.Context, config Config) error {
	ctx, cancel := context.WithCancel(parentCtx)
	defer cancel()

	// 1. Start Trickle server
	server := NewTrickleServer(config.Port)
	if err := server.Start(); err != nil {
		return fmt.Errorf("failed to start trickle server: %w", err)
	}
	defer server.Stop()

	// Give the server a moment to start
	time.Sleep(100 * time.Millisecond)

	// 2. Parse pipeline params
	var params map[string]interface{}
	if err := json.Unmarshal([]byte(config.Params), &params); err != nil {
		return fmt.Errorf("failed to parse params JSON: %w", err)
	}

	inputURL := server.GetInputURL()
	outputURL := server.GetOutputURL()
	controlURL := server.GetControlURL()
	eventsURL := server.GetEventsURL()

	// 3. Pre-create the input channel before calling runner API
	// This is critical - the runner will try to subscribe immediately
	slog.Info("Pre-creating trickle channels")
	if err := createTrickleChannel(inputURL); err != nil {
		slog.Warn("Failed to pre-create input channel", "err", err)
	}
	if err := createTrickleChannel(controlURL); err != nil {
		slog.Warn("Failed to pre-create control channel", "err", err)
	}

	// 4. Start output handler FIRST (subscribes to processed output)
	// Start before the runner so we're ready to receive output
	outputHandler := NewOutputHandler(outputURL, config.OutputFile)
	var wg sync.WaitGroup
	errCh := make(chan error, 2)

	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := outputHandler.Start(ctx); err != nil && ctx.Err() == nil {
			errCh <- fmt.Errorf("output handler error: %w", err)
		}
	}()

	// 5. Start segmenter in a goroutine (publishes input segments)
	// Start before calling runner API so input is available when runner connects
	segmenter := NewSegmenter(config.InputFile, inputURL, config.SegmentDur, config.Realtime)
	segmenterDone := make(chan struct{})

	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(segmenterDone)
		if err := segmenter.Start(ctx); err != nil && ctx.Err() == nil {
			errCh <- fmt.Errorf("segmenter error: %w", err)
		}
	}()

	// Give segmenter time to start publishing first segment
	time.Sleep(500 * time.Millisecond)

	// 6. Call AI Runner API to start the stream
	slog.Info("Starting stream on AI Runner",
		"inputURL", inputURL,
		"outputURL", outputURL,
		"runnerURL", config.RunnerURL,
	)

	if err := startStream(ctx, config.RunnerURL, inputURL, outputURL, controlURL, eventsURL, params); err != nil {
		return fmt.Errorf("failed to start stream on runner: %w", err)
	}

	// Wait for segmenter to finish, then give output handler time to receive remaining output
	go func() {
		<-segmenterDone
		slog.Info("Input finished, waiting for output to complete...")
		// Wait up to 30 seconds for output after input is done
		select {
		case <-time.After(30 * time.Second):
			slog.Info("Timeout waiting for output, stopping")
			cancel()
		case <-ctx.Done():
		}
	}()

	// Wait for completion or error
	doneCh := make(chan struct{})
	go func() {
		wg.Wait()
		close(doneCh)
	}()

	select {
	case err := <-errCh:
		return err
	case <-doneCh:
		return nil
	case <-ctx.Done():
		// Wait a bit for goroutines to clean up
		time.Sleep(500 * time.Millisecond)
		return nil
	}
}

// createTrickleChannel creates a channel on the trickle server
func createTrickleChannel(url string) error {
	req, err := http.NewRequest("POST", url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Expect-Content", "video/MP2T")

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// 200 or 404 (already exists) are both fine
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNotFound {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(body))
	}

	slog.Info("Created trickle channel", "url", url)
	return nil
}

func startStream(ctx context.Context, runnerURL, subscribeURL, publishURL, controlURL, eventsURL string, params map[string]interface{}) error {
	reqBody := StartStreamRequest{
		SubscribeURL: subscribeURL,
		PublishURL:   publishURL,
		ControlURL:   controlURL,
		EventsURL:    eventsURL,
		Params:       params,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/live-video-to-video", runnerURL)
	slog.Info("Calling AI Runner API", "url", url, "body", string(jsonBody))

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("runner returned status %d: %s", resp.StatusCode, string(body))
	}

	slog.Info("Stream started successfully", "response", string(body))
	return nil
}
