!pip3 install torch transformers einops

import torch
import transformers

model = transformers.AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b",trust_remote_code = True, torch_dtype = torch.bfloat16)

model.eval()
model.to("cuda:0")

