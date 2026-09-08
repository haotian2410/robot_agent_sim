from .pipeline.engine import PipelineEngine
def plan(instruction,**kwargs): return PipelineEngine().plan(instruction,**kwargs)
