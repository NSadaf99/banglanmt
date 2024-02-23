# import netrc
# import torch
# import torch.nn as nn
# import torch.optim as optim
# # Specify a path
# PATH = "state_dict_model.pt"

# # Save
# torch.save(netrc.state_dict(), PATH)

# # Load
# net = Net()
# model = Net()
# model.load_state_dict(torch.load(PATH))
# model.eval()

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from normalizer import normalize # pip install git+https://github.com/csebuetnlp/normalizer

model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/banglat5_nmt_bn_en")
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglat5_nmt_bn_en", use_fast=False)

input_sentence = "hi hi"
input_ids = tokenizer(normalize(input_sentence), return_tensors="pt").input_ids
generated_tokens = model.generate(input_ids)
decoded_tokens = tokenizer.batch_decode(generated_tokens)[0]

print(decoded_tokens)
