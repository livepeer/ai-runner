import logging
import os
import argparse
from huggingface_hub import snapshot_download
import re
import uvicorn
import json

logger = logging.getLogger(__name__)

MODELS_DIR = os.getenv("MODELS_DIR","/models")
PIPELINE = os.getenv("PIPELINE", "")
MODEL_ID = os.getenv("MODEL_ID", "")
HOST = os.getenv("HOST", "::")
PORT = os.getenv("PORT", "8000")

def get_download_cmd():
    include = []
    exclude = []
    with open('/app/dl_checkpoints.sh', 'r') as file:
        lines = file.readlines()  # Reads all lines into a list
        for line in lines:
            if MODEL_ID in line:
                line = line.strip()
                include_re = re.findall("--include\s+([\*\.\s\w\"]+)(?!\s--\w+)", line)
                exclude_re = re.findall("--exclude\s+([\*\.\s\w\"]+)(?!\s--\w+)", line)
                if len(include_re) > 0:
                    include = ["**/"+s.replace('"', '') for s in include_re[0].split()]
                if len(exclude_re) > 0:
                    exclude = ["**/"+s.replace('"', '') for s in exclude_re[0].split()]
                return include, exclude, MODELS_DIR
    
    #no automatic download found
    return [], [], ""

if __name__ == "__main__":
    allow, ignore, cache_folder = get_download_cmd()
    if len(allow) == 0 and len(ignore) == 0:
        logger.error("no include and exclude files found")
    
    #download the model
    print(f"Downloading model {MODEL_ID} from huggingface")
    snapshot_download(repo_id=MODEL_ID, cache_dir=cache_folder)

    print(f"Starting ai-runner")
    with open("/app/app/cfg/uvicorn_logging_config.json", "r") as file:
        log_config = json.load(file)

    #run the api
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=int(PORT),
        log_config=log_config
    )

