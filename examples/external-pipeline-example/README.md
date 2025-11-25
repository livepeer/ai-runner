# Example External Pipeline

This is a minimal example showing how to create an external pipeline package that can be installed separately from ai-runner.

## Structure

```
external-pipeline-example/
├── pyproject.toml          # Package definition with entry points
├── README.md
├── src/
│   └── example_pipeline/
│       ├── __init__.py
│       ├── pipeline.py      # Pipeline implementation
│       └── params.py        # Parameters definition
└── setup.py                 # Alternative to pyproject.toml
```

## Installation

```bash
# Install ai-runner-base first
cd /path/to/ai-runner/runner
pip install -e .

# Install this example pipeline
cd /path/to/external-pipeline-example
pip install -e .
```

## Usage

```bash
cd /path/to/ai-runner/runner
PIPELINE=example-pipeline MODEL_ID=example-pipeline python -m app.main
```

## Key Points

1. **Entry Points** - Registered in `pyproject.toml`:
   ```toml
   [project.entry-points."ai_runner.pipelines"]
   example-pipeline = "example_pipeline.pipeline:ExamplePipeline"
   ```

2. **Dependencies** - Depends on `ai-runner-base`:
   ```toml
   dependencies = ["ai-runner-base>=0.1.0"]
   ```

3. **Interface** - Implements `Pipeline` from `app.live.pipelines.interface`

4. **Multi-processing** - Works automatically with spawned subprocesses

