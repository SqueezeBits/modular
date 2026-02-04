
import sys
import os

# Add local max/python to sys.path
sys.path.append(os.getcwd() + "/max/python")

from max.pipelines.lib.registry import PIPELINE_REGISTRY
from max.pipelines.lib.hf_utils import HuggingFaceRepo, RepoType
from max.pipelines.lib.registry import PipelineRegistry

repo_path = "/home/jovyan/taesukim/models/FLUX.2-dev"
repo = HuggingFaceRepo(repo_id=repo_path)
print(f"Repo type: {repo.repo_type}")

print("Attempting to get active diffusers config...")
config = PIPELINE_REGISTRY.get_active_diffusers_config(repo)
if config:
    print("Success! Config found.")
    print(f"Class name: {config.get('_class_name')}")
else:
    print("Failed: Config is None.")
