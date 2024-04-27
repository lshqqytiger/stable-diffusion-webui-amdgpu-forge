import platform
import torch
from modules.sd_hijack_utils import CondFunc


memory_providers = ["None", "atiadlxx (AMD only)"]
default_memory_provider = "None"
if platform.system() == "Windows":
    memory_providers.append("Performance Counter")
    default_memory_provider = "Performance Counter"
do_nothing = lambda: None # pylint: disable=unnecessary-lambda-assignment
do_nothing_with_self = lambda self: None # pylint: disable=unnecessary-lambda-assignment


def _set_memory_provider():
    from modules.shared import opts, cmd_opts
    if opts.directml_memory_provider == "Performance Counter":
        from .backend import pdh_mem_get_info
        from .memory import MemoryProvider
        torch.dml.mem_get_info = pdh_mem_get_info
        if torch.dml.memory_provider is not None:
            del torch.dml.memory_provider
        torch.dml.memory_provider = MemoryProvider()
    elif opts.directml_memory_provider == "atiadlxx (AMD only)":
        device_name = torch.dml.get_device_name(cmd_opts.device_id)
        if "AMD" not in device_name and "Radeon" not in device_name:
            print(f"Memory stats provider is changed to None because the current device is not AMDGPU. Current Device: {device_name}")
            opts.directml_memory_provider = "None"
            _set_memory_provider()
            return
        from .backend import amd_mem_get_info
        torch.dml.mem_get_info = amd_mem_get_info
    else:
        from .backend import mem_get_info
        torch.dml.mem_get_info = mem_get_info
    torch.cuda.mem_get_info = torch.dml.mem_get_info


def directml_init():
    try:
        from modules.dml.backend import DirectML # pylint: disable=ungrouped-imports
        # Alternative of torch.cuda for DirectML.
        torch.dml = DirectML

        torch.cuda.is_available = lambda: False
        torch.cuda.device = torch.dml.device
        torch.cuda.device_count = torch.dml.device_count
        torch.cuda.current_device = torch.dml.current_device
        torch.cuda.get_device_name = torch.dml.get_device_name
        torch.cuda.get_device_properties = torch.dml.get_device_properties

        torch.cuda.empty_cache = do_nothing
        torch.cuda.ipc_collect = do_nothing
        torch.cuda.memory_stats = torch.dml.memory_stats
        torch.cuda.mem_get_info = torch.dml.mem_get_info
        torch.cuda.memory_allocated = torch.dml.memory_allocated
        torch.cuda.max_memory_allocated = torch.dml.max_memory_allocated
        torch.cuda.reset_peak_memory_stats = torch.dml.reset_peak_memory_stats
        torch.cuda.utilization = lambda: 0

        torch.Tensor.directml = lambda self: self.to(torch.dml.current_device())
    except Exception as e:
        print(f'DirectML initialization failed: {e}')
        return False, e
    return True, None


def directml_do_hijack():
    import modules.dml.hijack # noqa: F401
    from modules.devices import device

    CondFunc('torch.Generator',
        lambda orig_func, device: orig_func("cpu"),
        lambda orig_func, device: True)

    if not torch.dml.has_float64_support(device):
        torch.Tensor.__str__ = do_nothing_with_self
        CondFunc('torch.from_numpy',
            lambda orig_func, *args, **kwargs: orig_func(args[0].astype('float32')),
            lambda *args, **kwargs: args[1].dtype == float)

    _set_memory_provider()

def directml_override_opts():
    _set_memory_provider()
