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
      
answer_text = "This is another form of a text"
      
tokenized_answer = tokenizer.encode(answer_text, return_tensors = "pt")
      
outputs = model(**tokenized_example.to("cuda:0"))
print(outputs.logits.shape)
      
print(tokenized_example["input_ids"].shape)
      
last_token_output = outputs.logits[0,-1].view(1,-1)
print(last_token_output.shape())
      
tokenized_answer.shape()
      
loss_fct = torch.nn.CrossEntropyLoss()
optimizer = transformers.AdamW(model.parameters(), lr = 5e-5)
      
labels = tokenized_answer[0][0].view(1)
loss = loss_fct(last_token_output, label.to("cuda:0"))
      
print(loss.item)
      
model.train()
loss.backward
optimizer.step()
optimizer.zero_grad(set_to_none= True)      

