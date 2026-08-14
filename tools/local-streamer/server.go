package main

import (
	"fmt"
	"log/slog"
	"net/http"

	"github.com/livepeer/go-livepeer/trickle"
)

// TrickleServer wraps the trickle server functionality
type TrickleServer struct {
	server   *trickle.Server
	mux      *http.ServeMux
	httpSrv  *http.Server
	port     int
	basePath string
	stopFunc func()
}

// NewTrickleServer creates a new Trickle server on the specified port
func NewTrickleServer(port int) *TrickleServer {
	basePath := "/trickle/"
	mux := http.NewServeMux()

	config := trickle.TrickleServerConfig{
		BasePath:   basePath,
		Mux:        mux,
		Autocreate: true, // Auto-create channels on first publish
	}

	server := trickle.ConfigureServer(config)

	return &TrickleServer{
		server:   server,
		mux:      mux,
		port:     port,
		basePath: basePath,
	}
}

// Start starts the Trickle server
func (ts *TrickleServer) Start() error {
	// Start the trickle server's internal ticker for idle channel sweeping
	ts.stopFunc = ts.server.Start()

	addr := fmt.Sprintf(":%d", ts.port)
	ts.httpSrv = &http.Server{
		Addr:    addr,
		Handler: ts.mux,
	}

	slog.Info("Starting Trickle server", "addr", addr, "basePath", ts.basePath)

	go func() {
		if err := ts.httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("Trickle server error", "err", err)
		}
	}()

	return nil
}

// Stop stops the Trickle server
func (ts *TrickleServer) Stop() error {
	slog.Info("Stopping Trickle server")
	if ts.stopFunc != nil {
		ts.stopFunc()
	}
	if ts.httpSrv != nil {
		return ts.httpSrv.Close()
	}
	return nil
}

// GetInputURL returns the URL for the input channel
func (ts *TrickleServer) GetInputURL() string {
	return fmt.Sprintf("http://localhost:%d%sinput", ts.port, ts.basePath)
}

// GetOutputURL returns the URL for the output channel
func (ts *TrickleServer) GetOutputURL() string {
	return fmt.Sprintf("http://localhost:%d%soutput", ts.port, ts.basePath)
}

// GetControlURL returns the URL for the control channel
func (ts *TrickleServer) GetControlURL() string {
	return fmt.Sprintf("http://localhost:%d%scontrol", ts.port, ts.basePath)
}

// GetEventsURL returns the URL for the events channel
func (ts *TrickleServer) GetEventsURL() string {
	return fmt.Sprintf("http://localhost:%d%sevents", ts.port, ts.basePath)
}

