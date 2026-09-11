from .reference import ReferenceRenderer


def make_renderer(config):
    backend = config["backend"]
    if backend == "reference":
        return ReferenceRenderer(config.get("ray_chunk", 256), config.get("gaussian_chunk", 256), config.get("checkpoint", True))
    if backend == "cuda":
        from .cuda import CudaRenderer
        return CudaRenderer()
    raise ValueError(f"Unknown renderer: {backend}")
