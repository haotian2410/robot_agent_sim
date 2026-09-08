import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(os.environ.get("ROBOT_AGENT_SIM_ASSET_ROOT", PROJECT_ROOT / "assets"))
MODEL_ROOT = ASSET_ROOT / "models"
ROBOT_ROOT = ASSET_ROOT / "robots"
