package worker

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/docker/cli/opts"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/client"
	"github.com/docker/go-connections/nat"
)

const containerModelDir = "/models"
const containerPort = "8000/tcp"
const pollingInterval = 500 * time.Millisecond

var containerHostPorts = map[string]string{
	"text-to-image":  "8000",
	"image-to-image": "8001",
	"image-to-video": "8002",
}

type RunnerContainer struct {
	ID     string
	Client *ClientWithResponses
}

type Worker struct {
	containerImageID string
	gpus             string
	modelDir         string

	dockerClient *client.Client
	containers   map[string]*RunnerContainer
}

func NewWorker(containerImageID string, gpus string, modelDir string) (*Worker, error) {
	dockerClient, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, err
	}

	return &Worker{
		containerImageID: containerImageID,
		gpus:             gpus,
		modelDir:         modelDir,
		dockerClient:     dockerClient,
		containers:       make(map[string]*RunnerContainer),
	}, nil
}

func (w *Worker) TextToImage(ctx context.Context, modelID string, req TextToImageJSONRequestBody) ([]string, error) {
	c, err := w.getWarmContainer(ctx, "text-to-image", modelID)
	if err != nil {
		return nil, err
	}

	resp, err := c.Client.TextToImageWithResponse(ctx, req)
	if err != nil {
		return nil, err
	}

	if resp.JSON422 != nil {
		// TODO: Handle JSON422 struct
		return nil, errors.New("text-to-image container returned 422")
	}

	urls := make([]string, len(resp.JSON200.Images))
	for i, media := range resp.JSON200.Images {
		urls[i] = media.Url
	}

	return urls, nil
}

func (w *Worker) ImageToImage(ctx context.Context, modelID string, req ImageToImageMultipartRequestBody) ([]string, error) {
	return nil, nil
}

func (w *Worker) ImageToVideo(ctx context.Context, modelID string, req ImageToVideoMultipartRequestBody) ([]string, error) {
	return nil, nil
}

func (w *Worker) Warm(ctx context.Context, containerName, modelID string) error {
	_, err := w.getWarmContainer(ctx, containerName, modelID)
	return err
}

func (w *Worker) Stop(ctx context.Context, containerName string) error {
	c, ok := w.containers[containerName]
	if !ok {
		return fmt.Errorf("container %v is not running", containerName)
	}

	// TODO: Handle if container fails to stop
	delete(w.containers, containerName)

	return w.dockerClient.ContainerStop(ctx, c.ID, container.StopOptions{})
}

func (w *Worker) getWarmContainer(ctx context.Context, containerName string, modelID string) (*RunnerContainer, error) {
	c, ok := w.containers[containerName]
	if ok {
		return c, nil
	}

	// TODO: Pull image to ensure it exists

	containerConfig := &container.Config{
		Image: w.containerImageID,
		Env: []string{
			"PIPELINE=" + containerName,
			"MODEL_ID=" + modelID,
		},
		Volumes: map[string]struct{}{
			containerModelDir: {},
		},
		ExposedPorts: nat.PortSet{
			containerPort: struct{}{},
		},
	}

	gpuOpts := opts.GpuOpts{}
	gpuOpts.Set(w.gpus)

	containerHostPort := containerHostPorts[containerName]
	hostConfig := &container.HostConfig{
		Resources: container.Resources{
			DeviceRequests: gpuOpts.Value(),
		},
		Mounts: []mount.Mount{
			{
				Type:   mount.TypeBind,
				Source: w.modelDir,
				Target: containerModelDir,
			},
		},
		PortBindings: nat.PortMap{
			containerPort: []nat.PortBinding{
				{
					HostIP:   "0.0.0.0",
					HostPort: containerHostPort,
				},
			},
		},
	}

	resp, err := w.dockerClient.ContainerCreate(ctx, containerConfig, hostConfig, nil, nil, containerName)
	if err != nil {
		return nil, err
	}

	if err := w.dockerClient.ContainerStart(ctx, resp.ID, types.ContainerStartOptions{}); err != nil {
		return nil, err
	}

	// TODO: Add timeout with context
	if err := dockerWaitUntilRunning(ctx, w.dockerClient, resp.ID, pollingInterval); err != nil {
		return nil, err
	}

	client, err := NewClientWithResponses("http://localhost:" + containerHostPort)
	if err != nil {
		return nil, err
	}

	// TODO: Add timeout with context
	if err := runnerWaitUntilReady(ctx, client, pollingInterval); err != nil {
		return nil, err
	}

	c = &RunnerContainer{
		ID:     resp.ID,
		Client: client,
	}

	w.containers[containerName] = c

	return c, nil
}

func dockerWaitUntilRunning(ctx context.Context, client *client.Client, containerID string, pollingInterval time.Duration) error {
	ticker := time.NewTicker(pollingInterval)
	defer ticker.Stop()

tickerLoop:
	for range ticker.C {
		select {
		case <-ctx.Done():
			return errors.New("timed out waiting for container")
		default:
			json, err := client.ContainerInspect(ctx, containerID)
			if err != nil {
				return err
			}

			if json.State.Running {
				break tickerLoop
			}
		}
	}

	return nil
}

func runnerWaitUntilReady(ctx context.Context, client *ClientWithResponses, pollingInterval time.Duration) error {
	ticker := time.NewTicker(pollingInterval)
	defer ticker.Stop()

tickerLoop:
	for range ticker.C {
		select {
		case <-ctx.Done():
			return errors.New("timed out waiting for runner")
		default:
			if _, err := client.HealthWithResponse(ctx); err == nil {
				break tickerLoop
			}
		}
	}

	return nil
}
