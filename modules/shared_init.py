import os

import torch

from modules import shared
from modules.shared import cmd_opts
from modules.dml import directml_init, directml_do_hijack
from modules.zluda import initialize_zluda


def initialize():
    """Initializes fields inside the shared module in a controlled manner.

    Should be called early because some other modules you can import mingt need these fields to be already set.
    """

    os.makedirs(cmd_opts.hypernetwork_dir, exist_ok=True)

    from modules import options, shared_options
    shared.options_templates = shared_options.options_templates
    shared.opts = options.Options(shared_options.options_templates, shared_options.restricted_opts)
    shared.restricted_opts = shared_options.restricted_opts
    try:
        shared.opts.load(shared.config_filename)
    except FileNotFoundError:
        pass

    if cmd_opts.directml:
        directml_init()
        directml_do_hijack()
    else:
        torch.Tensor.__str__ = lambda self: None

    from modules import devices
    devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = \
        (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])

    devices.dtype = torch.float32 if cmd_opts.no_half else torch.float16
    devices.dtype_vae = torch.float32 if cmd_opts.no_half or cmd_opts.no_half_vae else torch.float16
    devices.dtype_inference = torch.float32 if cmd_opts.precision == 'full' else devices.dtype

    shared.device = devices.device
    shared.weight_load_location = None if cmd_opts.lowram else "cpu"

    from modules import shared_state
    shared.state = shared_state.State()
    shared.compiled_model_state = shared_state.CompiledModelState()

    from modules import styles
    shared.prompt_styles = styles.StyleDatabase(shared.styles_filename)

    from modules import interrogate
    shared.interrogator = interrogate.InterrogateModels("interrogate")

    from modules import shared_total_tqdm
    shared.total_tqdm = shared_total_tqdm.TotalTQDM()

    from modules import memmon, devices
    shared.mem_mon = memmon.MemUsageMonitor("MemMon", devices.device, shared.opts)
    shared.mem_mon.start()

    if not cmd_opts.skip_ort:
        from modules.onnx_impl import initialize_onnx
        initialize_onnx()

    initialize_zluda()
