"""Detector execution settings, independent of tracking and ROS."""


class RuntimeDetector:
    """Forward detector calls with explicit device and input-size settings."""

    def __init__(self, model, device, image_size=0):
        self.model, self.device = model, device
        self.image_size = int(image_size)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        kwargs["device"] = self.device
        if self.image_size:
            kwargs["imgsz"] = self.image_size
        if str(self.device).lower() == "cpu":
            # Remove the GPU-specific option for CPU inference.
            kwargs.pop("quantize", None)
        return self.model(*args, **kwargs)
