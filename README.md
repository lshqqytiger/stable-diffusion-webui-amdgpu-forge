# Stable Diffusion WebUI AMDGPU Forge

Stable Diffusion WebUI AMDGPU Forge is a platform on top of [Stable Diffusion WebUI AMDGPU](https://github.com/lshqqytiger/stable-diffusion-webui-amdgpu) (based on [Gradio](https://www.gradio.app/)) to make development easier, optimize resource management, speed up inference, and study experimental features.

The name "Forge" is inspired from "Minecraft Forge". This project is aimed at becoming SD WebUI AMDGPU's Forge.

Forge is currently based on SD-WebUI 1.10.1 at [this commit](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/82a973c04367123ae98bd9abdf80d9eda9b910e2).

# What's different from upstream repo?

This is a merge of [stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) and [stable-diffusion-webui-amdgpu](https://github.com/lshqqytiger/stable-diffusion-webui-amdgpu).

- `--zluda`: Use [ZLUDA](https://github.com/vosen/ZLUDA) as a torch backend.
- Support ONNX Runtime. (DirectML, CUDA, CPU)
- Support Olive model optimization. (DirectML, CUDA)

# Installing Forge

If you are proficient in Git and you want to install Forge as another branch of SD-WebUI, please see [here](https://github.com/continue-revolution/sd-webui-animatediff/blob/forge/master/docs/how-to-use.md#you-have-a1111-and-you-know-git). In this way, you can reuse all SD checkpoints and all extensions you installed previously in your OG SD-WebUI, but you should know what you are doing.

If you know what you are doing, you can install Forge using same method as SD-WebUI. (Install Git, Python, Git Clone the forge repo `https://github.com/lshqqytiger/stable-diffusion-webui-amdgpu-forge.git` and then run webui-user.bat).

![image](https://github.com/lshqqytiger/stable-diffusion-webui-amdgpu-forge/assets/19834515/c49bd60d-82bd-4086-9859-88d472582b94)

### Previous Versions

You can download previous versions [here](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/849).

# Forge Status

Based on manual test one-by-one:

| Component                                         | Status | Last Test    |
| ------------------------------------------------- | ------ | ------------ |
| Basic Diffusion                                   | Normal | 2024 July 27 |
| GPU Memory Management System                      | Normal | 2024 July 27 |
| LoRAs                                             | Normal | 2024 July 27 |
| All Preprocessors                                 | Normal | 2024 July 27 |
| All ControlNets                                   | Normal | 2024 July 27 |
| All IP-Adapters                                   | Normal | 2024 July 27 |
| All Instant-IDs                                   | Normal | 2024 July 27 |
| All Reference-only Methods                        | Normal | 2024 July 27 |
| All Integrated Extensions                         | Normal | 2024 July 27 |
| Popular Extensions (Adetailer, etc)               | Normal | 2024 July 27 |
| Gradio 4 UIs                                      | Normal | 2024 July 27 |
| Gradio 4 Forge Canvas                             | Normal | 2024 July 27 |
| LoRA/Checkpoint Selection UI for Gradio 4         | Normal | 2024 July 27 |
| Photopea/OpenposeEditor/etc for ControlNet        | Normal | 2024 July 27 |
| Wacom 128 level touch pressure support for Canvas | Normal | 2024 July 15 |

Feel free to open issue if anything is broken and I will take a look every several days. If I do not update this "Forge Status" then it means I cannot reproduce any problem. In that case, fresh re-install should help most.

# Under Construction

This Readme is under construction ... more docs/wiki coming soon ...
