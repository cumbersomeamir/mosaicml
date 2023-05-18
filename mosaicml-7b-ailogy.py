#!pip3 install torch transformers einops

import torch
import transformers

model = transformers.AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b",trust_remote_code = True, torch_dtype = torch.bfloat16)

model.eval()
model.to("cuda:0")

model_size = sum(t.numel() for t in model.parameters())
print(f"Model Size: {model_size/1000**2:.1f} M parameters")

tokenizer = transformers.AutoTokenizer.from_pretrained("mosaicml/mpt-7b")

txt = "This is a text to text mosaic"

tokenized_example = tokenizer(txt, return_tensors="pt")


print(tokenized_example['input_ids']
      
model.generate(tokenized_example['input_ids'].to('cuda:0'), max_new_tokens =150, do_sample = False, top_k = 5, top_p = 0.95)

answer = tokenizer.batch_decode(outputs, skip_special_tokens = True)
      
print(answer[0].rstrip())
      


