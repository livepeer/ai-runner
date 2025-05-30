# Development Documentation

This guide aims to assist developers working on the [AI runner](https://github.com/livepeer/ai-worker/tree/main/runner), offering detailed instructions for debugging and setting up the development environment. For general information about the AI runner, refer to the [AI Runner README](../README.md).

## Debugging

### Using the Provided DevContainer

Leverage the [VSCode DevContainer](https://code.visualstudio.com/docs/remote/containers) for an efficient debugging experience with the [AI runner](https://github.com/livepeer/ai-worker/tree/main/runner). This configuration automatically prepares a development environment equipped with all necessary tools and dependencies.

**Prerequisites:**

- [VSCode](https://code.visualstudio.com/download)
- [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

**Quickstart with DevContainer:**

1. **Install** [VSCode](https://code.visualstudio.com/download) and the [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
2. **Clone** the repository and open it in VSCode.
3. **Navigate** to the `runner` directory.
4. **Open in Container**: Click "Reopen in Container" when prompted, or manually initiate it by pressing `F1`, typing "Reopen in Container", and pressing `Enter`.
5. **Initialization**: The initial build may take a few minutes. Subsequent starts are faster.
6. **Begin Debugging**: The AI runner is now set up for debugging.

For more, see the [VSCode documentation on DevContainers](https://code.visualstudio.com/docs/devcontainers/containers).

### Debugging with External Services

To debug the AI runner when it operates within a container orchestrated by external services, such as [go-livepeer](https://github.com/livepeer/go-livepeer/tree/ai-video), follow these steps to attach to the container:

1. **Update AI Runner Module Path**: In the `go.mod` file of your local [go-livepeer](https://github.com/livepeer/go-livepeer/tree/ai-video) folder, modify the module path for the AI runner to point to your local version:

   ```bash
   go mod edit -replace github.com/livepeer/ai-worker=../path/to/ai-worker
   ```

2. **Directory**: Ensure you're in the `runner` directory.
3. **Build the AI Runner Image**:

   ```bash
   docker build --build-arg VERSION=$(./print_version.sh) -t livepeer/ai-runner:base .
   ```

4. **Build the Debug Image**:

   ```bash
   docker build -f ./dev/Dockerfile.debug -t livepeer/ai-runner:latest .
   ```

5. **Apply the Debug Patch**: Implement the required code modifications to facilitate debugger attachment and expose the necessary ports.

   ```bash
   cd .. && git apply ./runner/dev/patches/debug.patch && cd runner
   ```

6. **Attach and Debug**: Debugging the AI runner involves attaching to an active container. Ensure that VSCode is open in the `runner` directory. Follow these [instructions to attach to a running container](https://code.visualstudio.com/docs/python/debugging#_command-line-debugging) with the appropriate configuration:

   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Python Debugger: Remote Attach",
         "type": "debugpy",
         "request": "attach",
         "connect": {
           "host": "0.0.0.0",
           "port": 5678
         },
         "pathMappings": [
           {
             "localRoot": "${workspaceFolder}",
             "remoteRoot": "."
           }
         ]
       }
     ]
   }
   ```

7. **Revert Changes**: After debugging, undo the debug patch.

   ```bash
   cd .. && git apply -R ./runner/dev/patches/debug.patch && cd runner
   ```

8. **Rebuild the Image**: Execute a rebuild of the image, ensuring to exclude the debug changes.

   ```bash
   docker build --build-arg VERSION=$(./print_version.sh) -t livepeer/ai-runner:latest .
   ```

### Mocking the Pipelines

Mocking the pipelines is a practical approach for accelerating development and testing phases. This method simulates the pipeline execution, eliminating the need to run the actual model on a dedicated GPU. Follow the steps below to implement mocking:

1. **Navigate to the Correct Directory**:
   Ensure you are within the `runner` directory to apply changes effectively.

2. **Applying the Mock Patch**:
   Use the command below to apply the necessary code modifications for mocking the pipelines. This step introduces a mock environment for your development process.

   ```bash
   cd .. && git apply ./runner/dev/patches/mock.patch && cd runner
   ```

3. **Starting the AI Runner with Mocking**: Launch the AI runner with the environment variable `MOCK_PIPELINE` set to `True`. This enables the mock mode for pipeline execution.
4. **Reverting Mock Changes**: Once testing is complete and you wish to return to the actual pipeline execution, revert the applied mock changes using the following command:

   ```bash
   cd .. && git apply -R ./runner/dev/patches/mock.patch && cd runner
   ```
