import torch
import numpy as np
try:
    import onnxruntime as ort
except ImportError:
    ort = None


class OnnxPolicyWrapper:
    """Minimal wrapper so ONNXRuntime policies mimic TorchScript call signature.

    Supports single- or multi-input ONNX models. Call as
      policy(obs)                 # single-input
      policy(obs, hist, ...)      # multi-input, positional order matches ONNX inputs
    Inputs may be torch.Tensor or numpy arrays; output is returned as a torch.Tensor
    on CPU. `input_names` / `input_shapes` are exposed so callers can introspect
    (e.g. read the ONNX-declared history length).
    """

    def __init__(self, session, input_names, input_shapes, output_index=0):
        self.session = session
        self.input_names = list(input_names)
        self.input_shapes = list(input_shapes)
        self.output_index = output_index

    def __call__(self, *tensors) -> torch.Tensor:
        if len(tensors) != len(self.input_names):
            raise ValueError(
                f"Expected {len(self.input_names)} inputs ({self.input_names}), "
                f"got {len(tensors)}"
            )
        feed = {}
        for name, t in zip(self.input_names, tensors):
            if isinstance(t, torch.Tensor):
                arr = t.detach().cpu().numpy()
            else:
                arr = np.asarray(t)
            feed[name] = arr.astype(np.float32, copy=False)
        outputs = self.session.run(None, feed)
        result = outputs[self.output_index]
        if not isinstance(result, np.ndarray):
            result = np.asarray(result, dtype=np.float32)
        return torch.from_numpy(result.astype(np.float32))


def load_onnx_policy(policy_path: str, device: str) -> OnnxPolicyWrapper:
    if ort is None:
        raise ImportError("onnxruntime is required for ONNX policy inference but is not installed.")
    providers = []
    available = ort.get_available_providers()
    if device.startswith('cuda'):
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        else:
            print("CUDAExecutionProvider not available in onnxruntime; falling back to CPUExecutionProvider.")
    providers.append('CPUExecutionProvider')
    session = ort.InferenceSession(policy_path, providers=providers)
    inputs = session.get_inputs()
    input_names = [i.name for i in inputs]
    input_shapes = [i.shape for i in inputs]
    print(f"ONNX policy loaded from {policy_path} using providers: {session.get_providers()}")
    return OnnxPolicyWrapper(session, input_names, input_shapes)
