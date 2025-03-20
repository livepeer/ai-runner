# Realtime AI Dev Setup

1. Run MediaMTX locally:

    ```bash
    docker run --rm -d --network=host bluenviron/mediamtx:latest
    ```

2. Download/compile the live models to `~/.lpdData/models`

    ```bash
    cd ~/.lpData
    curl -s https://raw.githubusercontent.com/livepeer/ai-worker/main/runner/dl_checkpoints.sh | bash -s -- --tensorrt
    ```

3. No need to run the runner container manually anymore as the `ai-worker` has been fixed to manage them. Simply stopping the input stream will automatically kill the container.

    <aside>
    ⚠️

    In case you have issues starting the stream again, make sure the previous container was really killed (there might be issues in the logic). Use `docker ps` to look for any container like `live-video-to-video_PIPELINE_8900` and `rm -f` it manually.:q

    </aside>

4. Install ffmpeg from go-livepeer:

    ```bash
    export ROOT="$HOME/buildoutput"
    mkdir -p $ROOT/compiled
    export LD_LIBRARY_PATH="$ROOT/compiled/lib/"
    export PKG_CONFIG_PATH="$ROOT/compiled/lib/pkgconfig"
    export PATH=~/compiled/bin/:$PATH

    ./install_ffmpeg.sh ~/buildoutput

    # if you have issues installing with this script, simply use
    # curl https://raw.githubusercontent.com/livepeer/lpms/ffde2327537517b3345162e9544704571bc58a34/install_ffmpeg.sh | bash -s $1
    ```

5. Debug and launch go-livepeer orchestrator and gateway using the following scripts
    - **CUDA Video SDK on TensorDock**
        - If you are running on TensorDock, it is possible that you will get errors when running the Orchestrator due to lack of a couple video-related libs (e.g. `libnvcuvid.so.1`)
            - Try to run the orchestrator to see if you get the error, otherwise skip this section.
        - To fix it on my machine, I had to manually upgrade the CUDA version as well as the driver (maybe only the driver is enough, but why not both)
        - To upgrade CUDA, follow the corresponding Linux instructions [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network) (Ubuntu 22.04 x64). Or run these commands:

            ```go
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
            sudo dpkg -i cuda-keyring_1.1-1_all.deb
            sudo apt-get update
            sudo apt-get -y install cuda-toolkit-12-6
            ```

        - To upgrade the driver (maybe needs the `dpkg` above):

            ```go
            sudo apt-get install -y nvidia-driver-565
            sudo apt-get install -y cuda-drivers
            ```

            - (the `cuda-drivers` install directly complained about conflicts with the sub-package for driver so I had to install the `nvidia-driver-*` package manually first)
    - **run-ai-orch.sh**

        ```bash
        #!/bin/bash

        # Get primary local IP address
        IP_ADDR=$(hostname -I | awk '{print $1}')
        echo "Using service address: ${IP_ADDR}:8936"

        export GO111MODULE=ondo
        export CGO_ENABLED=1
        export CC=""
        export CGO_LDFLAGS="-L/usr/local/cuda/lib64 -L$HOME/buildoutput/compiled/lib -Wl,--copy-dt-needed-entries"
        export PATH="/usr/local/cuda/bin:$PATH"
        export PKG_CONFIG_PATH="$HOME/buildoutput/compiled/lib/pkgconfig"
        export LD_LIBRARY_PATH="$HOME/buildoutput/compiled/lib"

        go run -tags mainnet,experimental ./cmd/livepeer \
        	  -orchestrator \
        	  -dataDir "$HOME/.lpData/offchain" \
        	  -transcoder \
        	  -serviceAddr ${IP_ADDR}:8936 \
        	  -nvidia all \
        	  -aiWorker \
        	  -aiModels "$HOME/.lpData/aiModels-live-to-live.json" \
        	  -aiModelsDir "$HOME/.lpData/models" \
        	  -aiRunnerImage livepeer/ai-runner:latest \
        	  -cliAddr 127.0.0.1:7934 \
        	  -v 7

        ```

    - **run-ai-gtw.sh**

        ```bash
        #!/bin/bash

        # Get primary local IP address
        IP_ADDR=$(hostname -I | awk '{print $1}')
        echo "Using HTTP address: ${IP_ADDR}:8937"

        export GO111MODULE=on
        export CGO_ENABLED=1
        export CC=""
        export CGO_LDFLAGS="-L/usr/local/cuda/lib64 -L$HOME/buildoutput/compiled/lib -Wl,--copy-dt-needed-entries"
        export PATH="/usr/local/cuda/bin:$PATH"
        export PKG_CONFIG_PATH="$HOME/buildoutput/compiled/lib/pkgconfig"
        export LD_LIBRARY_PATH="$HOME/buildoutput/compiled/lib"

        go run -tags mainnet,experimental ./cmd/livepeer \
        	-gateway \
        	-dataDir "$HOME/.lpData/offchain" \
        	-httpAddr ${IP_ADDR}:8937 \
        	-httpIngest \
        	-rtmpAddr 0.0.0.0:1933 \
        	-cliAddr 127.0.0.1:7951 \
        	-monitor \
        	-orchAddr https://${IP_ADDR}:8936 \
        	-v 6

        # docker run --rm -it \
        #     --name livepeer_ai_gateway \
        #     -v "$HOME/.lpData/:/root/.lpData" \
        #     -p 8937:8937 \
        #     --network host \
        #     livepeer/go-livepeer:master \
        #     -datadir /root/.lpData/offchain \
        #     -gateway \
        #     -httpAddr 192.168.122.198:8937 \
        # 		-rtmpAddr 0.0.0.0:1933 \
        # 		-cliAddr 0.0.0.0:7951 \
        # 		-httpIngest \
        # 		-orchAddr https://192.168.122.198:8936 \
        # 		-monitor \
        #     -v 6

        ```

    - Alternative (instead of above scripts): **launch.json**

        ```json
        {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "AI Core (offchain)",
                    "type": "go",
                    "request": "launch",
                    "mode": "auto",
                    "program": "${workspaceFolder}/go-livepeer/cmd/livepeer",
                    "buildFlags": "-tags=mainnet,experimental",
                    "env": {
                        "GO111MODULE": "on",
                        "CGO_ENABLED": "1",
                        "CC": "",
                        "CGO_LDFLAGS": "-L/usr/local/cuda/lib64 -L/home/user/buildoutput/compiled/lib -Wl,--copy-dt-needed-entries",
                        "PATH: ": "/usr/local/cuda/bin:${PATH}",
                        "PKG_CONFIG_PATH": "/home/user/buildoutput/compiled/lib/pkgconfig",
                        "LD_LIBRARY_PATH": "/home/user/buildoutput/compiled/lib"
                    },
                    "args": [
                        "-orchestrator",
                        "-dataDir",
                        "/home/user/.lpData/offchain",
                        "-transcoder",
                        "-serviceAddr",
                        "0.0.0.0:8936",
                        "-nvidia",
                        "all",
                        "-aiWorker",
                        "-aiModels",
                        "/home/user/.lpData/aiModels-live-to-live.json",
                        "-aiModelsDir",
                        "/home/user/.lpData/models",
                        "-aiRunnerImage",
                        "livepeer/ai-runner:latest",
                        "-cliAddr",
                        "127.0.0.1:7934",
                        "-v",
                        "7",
                    ],
                },
                {
                    "name": "AI Gateway (offchain)",
                    "type": "go",
                    "request": "launch",
                    "mode": "auto",
                    "program": "${workspaceFolder}/go-livepeer/cmd/livepeer",
                    "buildFlags": "-tags=mainnet,experimental",
                    "env": {
                        "GO111MODULE": "on",
                        "CGO_ENABLED": "1",
                        "CC": "",
                        "CGO_LDFLAGS": "-L/usr/local/cuda/lib64 -L/home/user/buildoutput/compiled/lib -Wl,--copy-dt-needed-entries",
                        "PATH: ": "/usr/local/cuda/bin:${PATH}",
                        "PKG_CONFIG_PATH": "/home/user/buildoutput/compiled/lib/pkgconfig",
                        "LD_LIBRARY_PATH": "/home/user/buildoutput/compiled/lib"
                    },
                    "args": [
                        "-gateway",
                        "-dataDir",
                        "/home/user/.lpData/offchain",
                        "-httpAddr",
                        "0.0.0.0:8937",
                        "-httpIngest",
                        "-rtmpAddr",
                        "0.0.0.0:1933",
                        "-cliAddr",
                        "127.0.0.1:7951",
                        "-monitor",
                        "-orchAddr",
                        "https://0.0.0.0:8936",
                        "-v",
                        "6"
                    ],
                },
            ],
            "compounds": [
                {
                    "name": "AI Offchain (G/W)",
                    "configurations": [
                        "AI Gateway (offchain)",
                        "AI Core (offchain)"
                    ]
                },
                {
                    "name": "AI Onchain (G/W)",
                    "configurations": [
                        "AI Gateway (onchain)",
                        "AI Core (onchain)"
                    ]
                }
            ]
        }
        ```

    - In case the `IP_ADDR` in the script doesn’t work, update it to be to your network iface IP so the dockerized runner can reach the O on the host.
    - Update the `ffmpeg` install path on `CGO_LDFLAGS` or install FFMPEG as instructed [above](https://www.notion.so/Realtime-AI-Dev-Setup-1460a34856878011bf44f25a547b4a63?pvs=21)
6. Create the `/.lpData/aiModels-live-to-live.json` file to point to your local runner

    ```bash
    [
            {
                    "pipeline": "live-video-to-video",
                    "model_id": "streamdiffusion",
                    "warm": true
            }
    ]
    ```

7. Start a stream to MediaMTX using OBS or `ffmpeg`. Example: `rtmp://localhost:1935/test`
    1. e.g. for `ffmpeg`

    ```go
    ffmpeg -stream_loop -1 -re -i ./bbb.mp4 -c copy -f flv rtmp://localhost:1935/test
    ```

8. Using `test` as the stream name, send a post request to the gateway:

    ```bash
    curl --location 'http://192.168.10.61:8937/live/video-to-video/test/start' \
    --form 'stream="test"' \
    --form 'source_id="testout"' \
    --form 'source_type="webrtcSession"' \
    --form 'query="pipeline=streamdiffusion&rtmpOutput=rtmp://rtmp.livepeer.com/live/0974-7x87-ha7l-i2yv"'
    ```

    - Be sure to set the IP, pipeline and rtmpOutput correctly in the above command. Stream name `test` is in URL path
    - Playback that example stream on [https://lvpr.tv?v=09741pxx08eab03n](https://lvpr.tv/?v=09741pxx08eab03n) (but probably create your own stream for testing)
9. Once the orchestrator is selected, an `ai-runner` container will be started and the request will be sent to it.
    - AI inference will only begin if the transcoding and connection to `rtmpOutput` is successful.
    - When rtmpOutput is omitted from the request, the local MediaMTX server is used and AI inference occurs on the runner, however the subscriber channel back from the ai-runner still fails to save/transmux the segments.

    <aside>
    👀

    If you want to see the E2E pipeline working and are getting the `ffmpeg` error below, you can use this branch to use the `ffmpeg` installed in your system instead of LPMS: ‣

    Notice that it doesn’t work on the containerized `go-livepeer` tho.

    </aside>
