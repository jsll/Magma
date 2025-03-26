import torch

from magma.modeling_magma import MagmaForCausalLM
from magma.processing_magma import MagmaProcessor

dtype = torch.bfloat16
model = MagmaForCausalLM.from_pretrained(
    "microsoft/Magma-8B",
    trust_remote_code=True,
    torch_dtype=dtype,
)
processor = MagmaProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)
model.to("cuda")
