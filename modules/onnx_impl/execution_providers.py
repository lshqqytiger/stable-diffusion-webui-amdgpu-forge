import sys
from enum import Enum
from typing import Tuple, List
import onnxruntime as ort
from modules import devices


class ExecutionProvider(str, Enum):
    CPU = "CPUExecutionProvider"
    DirectML = "DmlExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    ROCm = "ROCMExecutionProvider"
    MIGraphX = "MIGraphXExecutionProvider"
    OpenVINO = "OpenVINOExecutionProvider"


available_execution_providers: List[ExecutionProvider] = ort.get_available_providers()
EP_TO_NAME = {
    ExecutionProvider.CPU: "gpu-cpu", # ???
    ExecutionProvider.DirectML: "gpu-dml",
    ExecutionProvider.CUDA: "gpu-cuda", # test required
    ExecutionProvider.ROCm: "gpu-rocm", # test required
    ExecutionProvider.MIGraphX: "gpu-migraphx", # test required
}
TORCH_DEVICE_TO_EP = {
    "cpu": ExecutionProvider.CPU,
    "cuda": ExecutionProvider.CUDA,
    "privateuseone": ExecutionProvider.DirectML,
    "meta": None,
}


def get_default_execution_provider() -> ExecutionProvider:
    if devices.backend == "cpu":
        return ExecutionProvider.CPU
    elif devices.backend == "directml":
        return ExecutionProvider.DirectML
    elif devices.backend == "cuda":
        return ExecutionProvider.CUDA
    elif devices.backend == "rocm":
        return ExecutionProvider.ROCm
    elif devices.backend == "ipex" or devices.backend == "openvino":
        return ExecutionProvider.OpenVINO
    return ExecutionProvider.CPU


def get_execution_provider_options():
    from modules.shared import cmd_opts, opts
    execution_provider_options = { "device_id": int(cmd_opts.device_id or 0) }
    if opts.onnx_execution_provider == ExecutionProvider.ROCm:
        if ExecutionProvider.ROCm in available_execution_providers:
            execution_provider_options["tunable_op_enable"] = 1
            execution_provider_options["tunable_op_tuning_enable"] = 1
    return execution_provider_options


def get_provider() -> Tuple:
    from modules.shared import opts
    return (opts.onnx_execution_provider, get_execution_provider_options(),)


def install_execution_provider(ep: ExecutionProvider):
    from modules.launch_utils import is_installed, run_pip, run_pip_uninstall

    if is_installed("onnxruntime"):
        run_pip_uninstall("onnxruntime")
    if is_installed("onnxruntime-directml"):
        run_pip_uninstall("onnxruntime-directml")
    if is_installed("onnxruntime-gpu"):
        run_pip_uninstall("onnxruntime-gpu")
    if is_installed("onnxruntime-training"):
        run_pip_uninstall("onnxruntime-training")
    if is_installed("onnxruntime-openvino"):
        run_pip_uninstall("onnxruntime-openvino")

    packages = ["onnxruntime"] # Failed to load olive: cannot import name '__version__' from 'onnxruntime'

    if ep == ExecutionProvider.DirectML:
        packages.append("onnxruntime-directml")
    elif ep == ExecutionProvider.CUDA:
        packages.append("onnxruntime-gpu")
    elif ep == ExecutionProvider.ROCm:
        if "linux" not in sys.platform:
            print("ROCMExecutionProvider is not supported on Windows.")
            return

        packages.append("--pre onnxruntime-training --index-url https://pypi.lsh.sh/60 --extra-index-url https://pypi.org/simple")
    elif ep == ExecutionProvider.OpenVINO:
        if is_installed("openvino"):
            run_pip_uninstall("openvino")
        packages.append("openvino")
        packages.append("onnxruntime-openvino")

    run_pip(f"install --upgrade {' '.join(packages)}")
    print("Please restart SD.Next.")
