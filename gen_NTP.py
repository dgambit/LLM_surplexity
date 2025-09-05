from datasets import Dataset, load_dataset
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from unsloth import FastLanguageModel
from huggingface_hub import login
import torch

nsent = 1000
ntop = 100
model_zero = "llm2"
#dataset = "WXS"
ndoc = 1000
synt = 64
#filter = "LANG"
lr = "1e-04"

tok = 64
run = 0

tasks = ["wiki", "xlsum", "sciabs"]

#pipeline = "WXS_doc1000_synt64_lr1e-04_acm"
#pipeline = "doc1000_synt64_rnd42_lr5e-05_acm_SYNLAST"

device = torch.device("cuda:0")

synt=64

dsx ="WXS"

for model_zero in ["llm2"]:
    for filter in ["SYNLAST", "FRESH", "MPP", "LANG"]:
        pipeline = f"doc1000_synt{synt}_lr1e-04_acm"

        NTP_dict = {t: {gen:[] for gen in range(11)} for t in tasks}
        for gen in range(11):
            
            model_name = f"M_{model_zero}_run{run}_gen{gen}_{dsx}_{pipeline}_{filter}"
            print(model_name)
            model_path = "dgambettaphd/" + model_name
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_path,
                max_seq_length = 512,
                dtype = None,
                load_in_4bit = True,
            device_map = {"": 1}  )
            
            for task in tasks:

                ds = load_dataset(f"dgambettaphd/{task}", split = "train")
                sents = ds["text"][10000:10000+nsent]
                sents = [" ".join(s.split(" ")[:tok]) for s in sents]

                for input_text in sents:

                    encodings = tokenizer(input_text, return_tensors="pt")
                    input_ids = encodings["input_ids"].to(model.device)

                    with torch.no_grad():
                        outputs = model(input_ids)
                        logits = outputs.logits  

                    last_token_logits = logits[:, -1, :] 
                    probs = torch.nn.functional.softmax(last_token_logits, dim=-1)

                    top_probs, top_token_ids = torch.topk(probs, ntop, dim=-1)

                    top_tokens = [tokenizer.decode([token_id]) for token_id in top_token_ids[0]]
                    top_probs = top_probs[0].tolist()

                    NTP_dict[task][gen].append({"tokens": top_tokens, "probs": top_probs})

        file_path = f"stats_acm/NTP/"+model_name.replace(f"_gen{gen}_", "_")+"/"
        os.makedirs(file_path , exist_ok=True)

        with open(file_path + f'sent{nsent}_top{ntop}_tok{tok}.json', 'w') as fp:
            json.dump(NTP_dict, fp)
