from pathlib import Path
class SceneManager:
    def __init__(self, backend=None): self.backend=backend
    def load_uploaded(self,path:Path,robot:str):
        if path.suffix.lower() not in (".xml",".mjcf"): raise ValueError("scene must be XML/MJCF")
        return self.backend.load_uploaded(path,robot) if self.backend else {"scene_id":path.stem,"robot":robot}

    # ``load`` was the name used by the first skeleton.  Keep it as a small
    # compatibility alias while routing all real work through the explicit
    # uploaded-scene API.
    def load(self, path: Path, robot: str):
        return self.load_uploaded(path, robot)
