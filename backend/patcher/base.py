# Model Patching System, Copyright Forge 2024

# API Templates partially extracted From ComfyUI, the actual implementation for those APIs
# are from Forge, implemented from scratch (after forge-v1.0.1), and may have
# certain level of differences.

import time
import torch
import copy
import inspect

from tqdm import tqdm
from backend import memory_management, utils, operations
from backend.patcher.lora import merge_lora_to_model_weight


def set_model_options_patch_replace(model_options, patch, name, block_name, number, transformer_index=None):
    to = model_options["transformer_options"].copy()

    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if name not in to["patches_replace"]:
        to["patches_replace"][name] = {}
    else:
        to["patches_replace"][name] = to["patches_replace"][name].copy()

    if transformer_index is not None:
        block = (block_name, number, transformer_index)
    else:
        block = (block_name, number)
    to["patches_replace"][name][block] = patch
    model_options["transformer_options"] = to
    return model_options


def set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=False):
    model_options["sampler_post_cfg_function"] = model_options.get("sampler_post_cfg_function", []) + [post_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options


def set_model_options_pre_cfg_function(model_options, pre_cfg_function, disable_cfg1_optimization=False):
    model_options["sampler_pre_cfg_function"] = model_options.get("sampler_pre_cfg_function", []) + [pre_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options


class ModelPatcher:
    def __init__(self, model, load_device, offload_device, size=0, current_device=None, **kwargs):
        self.size = size
        self.model = model
        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.model_options = {"transformer_options": {}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        if current_device is None:
            self.current_device = self.offload_device
        else:
            self.current_device = current_device

    def model_size(self):
        if self.size > 0:
            return self.size
        self.size = memory_management.module_size(self.model)
        return self.size

    def clone(self):
        n = ModelPatcher(self.model, self.load_device, self.offload_device, self.size, self.current_device)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        return n

    def is_clone(self, other):
        if hasattr(other, 'model') and self.model is other.model:
            return True
        return False

    def memory_required(self, input_shape):
        return self.model.memory_required(input_shape=input_shape)

    def set_model_sampler_cfg_function(self, sampler_cfg_function, disable_cfg1_optimization=False):
        if len(inspect.signature(sampler_cfg_function).parameters) == 3:
            self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(args["cond"], args["uncond"], args["cond_scale"])  # Old way
        else:
            self.model_options["sampler_cfg_function"] = sampler_cfg_function
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_post_cfg_function(self, post_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_post_cfg_function(self.model_options, post_cfg_function, disable_cfg1_optimization)

    def set_model_sampler_pre_cfg_function(self, pre_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_pre_cfg_function(self.model_options, pre_cfg_function, disable_cfg1_optimization)

    def set_model_unet_function_wrapper(self, unet_wrapper_function):
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_vae_encode_wrapper(self, wrapper_function):
        self.model_options["model_vae_encode_wrapper"] = wrapper_function

    def set_model_vae_decode_wrapper(self, wrapper_function):
        self.model_options["model_vae_decode_wrapper"] = wrapper_function

    def set_model_vae_regulation(self, vae_regulation):
        self.model_options["model_vae_regulation"] = vae_regulation

    def set_model_denoise_mask_function(self, denoise_mask_function):
        self.model_options["denoise_mask_function"] = denoise_mask_function

    def set_model_patch(self, patch, name):
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        self.model_options = set_model_options_patch_replace(self.model_options, patch, name, block_name, number, transformer_index=transformer_index)

    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn1_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn1", block_name, number, transformer_index)

    def set_model_attn2_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn2", block_name, number, transformer_index)

    def set_model_attn1_output_patch(self, patch):
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch):
        self.set_model_patch(patch, "attn2_output_patch")

    def set_model_input_block_patch(self, patch):
        self.set_model_patch(patch, "input_block_patch")

    def set_model_input_block_patch_after_skip(self, patch):
        self.set_model_patch(patch, "input_block_patch_after_skip")

    def set_model_output_block_patch(self, patch):
        self.set_model_patch(patch, "output_block_patch")

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        else:
            if name in self.object_patches_backup:
                return self.object_patches_backup[name]
            else:
                return utils.get_attr(self.model, name)

    def model_patches_to(self, device):
        to = self.model_options["transformer_options"]
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], "to"):
                        patch_list[i] = patch_list[i].to(device)
        if "patches_replace" in to:
            patches = to["patches_replace"]
            for name in patches:
                patch_list = patches[name]
                for k in patch_list:
                    if hasattr(patch_list[k], "to"):
                        patch_list[k] = patch_list[k].to(device)
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_dtype(self):
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        p = set()
        model_sd = self.model.state_dict()
        for k in patches:
            offset = None
            function = None
            if isinstance(k, str):
                key = k
            else:
                offset = k[1]
                key = k[0]
                if len(k) > 2:
                    function = k[2]

            if key in model_sd:
                p.add(k)
                current_patches = self.patches.get(key, [])
                current_patches.append((strength_patch, patches[k], strength_model, offset, function))
                self.patches[key] = current_patches

        return list(p)

    def get_key_patches(self, filter_prefix=None):
        memory_management.unload_model_clones(self)
        model_sd = self.model_state_dict()
        p = {}
        for k in model_sd:
            if filter_prefix is not None:
                if not k.startswith(filter_prefix):
                    continue
            if k in self.patches:
                p[k] = [model_sd[k]] + self.patches[k]
            else:
                p[k] = (model_sd[k],)
        return p

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        keys = list(sd.keys())
        if filter_prefix is not None:
            for k in keys:
                if not k.startswith(filter_prefix):
                    sd.pop(k)
        return sd

    def forge_patch_model(self, target_device=None):
        execution_start_time = time.perf_counter()

        for k, item in self.object_patches.items():
            old = utils.get_attr(self.model, k)

            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old

            utils.set_attr_raw(self.model, k, item)

        for key, current_patches in (tqdm(self.patches.items(), desc='Patching LoRAs') if len(self.patches) > 0 else self.patches):
            try:
                weight = utils.get_attr(self.model, key)
                assert isinstance(weight, torch.nn.Parameter)
            except:
                raise ValueError(f"Wrong LoRA Key: {key}")

            if key not in self.backup:
                self.backup[key] = weight.to(device=self.offload_device)

            bnb_layer = None

            if operations.bnb_avaliable:
                if hasattr(weight, 'bnb_quantized'):
                    assert weight.module is not None, 'BNB bad weight without parent layer!'
                    bnb_layer = weight.module
                    if weight.bnb_quantized:
                        weight_original_device = weight.device

                        if target_device is not None:
                            assert target_device.type == 'cuda', 'BNB Must use CUDA!'
                            weight = weight.to(target_device)
                        else:
                            weight = weight.cuda()

                        from backend.operations_bnb import functional_dequantize_4bit
                        weight = functional_dequantize_4bit(weight)

                        if target_device is None:
                            weight = weight.to(device=weight_original_device)
                    else:
                        weight = weight.data

            if target_device is not None:
                weight = weight.to(device=target_device)

            gguf_cls, gguf_type, gguf_real_shape = None, None, None

            if hasattr(weight, 'is_gguf'):
                from backend.operations_gguf import dequantize_tensor
                gguf_cls = weight.gguf_cls
                gguf_type = weight.gguf_type
                gguf_real_shape = weight.gguf_real_shape
                weight = dequantize_tensor(weight)

            weight_original_dtype = weight.dtype
            weight = weight.to(dtype=torch.float32)
            weight = merge_lora_to_model_weight(current_patches, weight, key).to(dtype=weight_original_dtype)

            if bnb_layer is not None:
                bnb_layer.reload_weight(weight)
                continue

            if gguf_cls is not None:
                from backend.operations_gguf import ParameterGGUF
                weight = gguf_cls.quantize_pytorch(weight, gguf_real_shape)
                utils.set_attr_raw(self.model, key, ParameterGGUF.make(
                    data=weight,
                    gguf_type=gguf_type,
                    gguf_cls=gguf_cls,
                    gguf_real_shape=gguf_real_shape
                ))
                continue

            utils.set_attr_raw(self.model, key, torch.nn.Parameter(weight, requires_grad=False))

        if target_device is not None:
            self.model.to(target_device)
            self.current_device = target_device

        moving_time = time.perf_counter() - execution_start_time

        if moving_time > 0.1:
            print(f'LoRA patching has taken {moving_time:.2f} seconds')

        return self.model

    def forge_unpatch_model(self, target_device=None):
        keys = list(self.backup.keys())

        for k in keys:
            w = self.backup[k]

            if not isinstance(w, torch.nn.Parameter):
                # In very few cases
                w = torch.nn.Parameter(w, requires_grad=False)

            utils.set_attr_raw(self.model, k, w)

        self.backup = {}

        if target_device is not None:
            self.model.to(target_device)
            self.current_device = target_device

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            utils.set_attr_raw(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup = {}
        return
