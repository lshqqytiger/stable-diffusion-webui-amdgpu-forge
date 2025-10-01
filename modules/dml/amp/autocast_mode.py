import importlib
from typing import Any, Optional
import torch
import functools

ops = ["torch.Tensor.__matmul__", "torch.addbmm", "torch.addmm", "torch.addmv", "torch.addr", "torch.baddbmm", "torch.bmm", "torch.chain_matmul", "torch.linalg.multi_dot", "torch.nn.functional.conv1d", "torch.nn.functional.conv2d", "torch.nn.functional.conv3d", "torch.nn.functional.conv_transpose1d", "torch.nn.functional.conv_transpose2d", "torch.nn.functional.conv_transpose3d", "torch.nn.GRUCell", "torch.nn.functional.linear", "torch.nn.LSTMCell", "torch.matmul", "torch.mm", "torch.mv", "torch.prelu", "torch.nn.RNNCell", "torch.embedding"]
supported_cast_pairs = {
    torch.float16: (torch.float32,),
    torch.float32: (torch.float16,),
}

def _cast_tensor(tensor: torch.Tensor):
    """Casts a tensor to the autocast dtype if applicable."""
    if not torch.is_tensor(tensor):
        return tensor
    dtype: torch.dtype = tensor.dtype
    if dtype not in supported_cast_pairs or (torch.dml.autocast_gpu_dtype != dtype and torch.dml.autocast_gpu_dtype not in supported_cast_pairs[dtype]):
        return tensor
    return tensor.type(torch.dml.autocast_gpu_dtype)

# This provides a stable object that the PyTorch JIT compiler can inspect.
class _AutocastForwarder:
    def __init__(self, op):
        self.op = op
        # Copying metadata helps with introspection and makes the wrapper
        # more transparent to other tools, including the JIT compiler.
        functools.update_wrapper(self, op)

    def __call__(self, *args, **kwargs):
        if not torch.dml.is_autocast_enabled:
            return self.op(*args, **kwargs)
        
        # Cast all tensor arguments to the autocast dtype
        args = list(map(_cast_tensor, args))
        for kwarg in kwargs:
            kwargs[kwarg] = _cast_tensor(kwargs[kwarg])
            
        return self.op(*args, **kwargs)

def _patch_op(op_str: str):
    """Dynamically finds and patches the specified PyTorch operation."""
    if isinstance(op_str, str):
        func_path = op_str.split('.')
        # handles both functions (e.g., torch.matmul)
        # and class methods (e.g., torch.Tensor.__matmul__)
        for i in range(len(func_path)-1, -1, -1):
            try:
                module_path = '.'.join(func_path[:i])
                if module_path:
                    resolved_obj = importlib.import_module(module_path)
                    break
            except ImportError:
                pass
        else:
            # built-in types which don't have a module
            resolved_obj = __builtins__

        for attr_name in func_path[i:-1]:
            resolved_obj = getattr(resolved_obj, attr_name)
            
        original_op = getattr(resolved_obj, func_path[-1])  
        setattr(resolved_obj, func_path[-1], _AutocastForwarder(original_op))

for o in ops:
    try:
        _patch_op(o)
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not patch {o}. Reason: {e}")


class autocast:
    prev: bool

    fast_dtype: torch.dtype = torch.float16
    prev_fast_dtype: torch.dtype
    def __init__(self, dtype: Optional[torch.dtype] = torch.float16):
        self.fast_dtype = dtype

    def __enter__(self):
        self.prev = torch.dml.is_autocast_enabled
        self.prev_fast_dtype = torch.dml.autocast_gpu_dtype
        torch.dml.is_autocast_enabled = True
        torch.dml.autocast_gpu_dtype = self.fast_dtype

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        torch.dml.is_autocast_enabled = self.prev
        torch.dml.autocast_gpu_dtype = self.prev_fast_dtype
