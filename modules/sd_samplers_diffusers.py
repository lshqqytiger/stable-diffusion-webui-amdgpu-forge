import diffusers
from modules.sd_samplers_common import SamplerData

samplers = [
    SamplerData("PNDM", diffusers.PNDMScheduler, [], None),
    SamplerData("LMS", diffusers.LMSDiscreteScheduler, [], None),
    SamplerData("Heun", diffusers.HeunDiscreteScheduler, [], None),
    SamplerData("DDIM", diffusers.DDIMScheduler, [], None),
    SamplerData("DDPM", diffusers.DDPMScheduler, [], None),
    SamplerData("Euler", diffusers.EulerDiscreteScheduler, [], None),
    SamplerData("Euler a", diffusers.EulerAncestralDiscreteScheduler, [], None),
    SamplerData("DPM", diffusers.DPMSolverMultistepScheduler, [], None),
]
