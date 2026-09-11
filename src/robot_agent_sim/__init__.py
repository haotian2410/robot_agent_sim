__all__ = ["PipelineEngine", "PipelineResult"]


def __getattr__(name: str):
    """Keep the public API while avoiding an eager MuJoCo/GL import."""
    if name in __all__:
        from .pipeline.engine import PipelineEngine, PipelineResult

        return {"PipelineEngine": PipelineEngine, "PipelineResult": PipelineResult}[name]
    raise AttributeError(name)
