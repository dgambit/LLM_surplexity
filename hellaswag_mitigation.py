import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import tasks, evaluator
import lm_eval
import os
import json
import time
from huggingface_hub import hf_hub_url, model_info

def wait_until_model_exists(model_name):
    while True:
        try:
            model_info(model_name)   # ← prova a cercare il modello
            print(f"Modello {model_name} trovato!")
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, 
                                        device_map=None).to(device)  
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            results =  evaluator.simple_evaluate(model = lm_eval.models.huggingface.HFLM(pretrained=model),
                                tasks =["hellaswag"])

            return results                 # ← esce dal while e dalla funzione
        except:
            print(f"Modello {model_name} non ancora disponibile. Aspetto 5 minuti...")
            time.sleep(900)        # ← attende 5 minuti prima di riprovare


labels = []


labels += [f"M_llm2_run1_gen{gen}_W_doc1000_synt64_lr1e-04_acm_SYNLAST" for gen in range(11)]
labels += [f"M_llm2_run1_gen{gen}_X_doc1000_synt64_lr1e-04_acm_SYNLAST" for gen in range(11)]
labels += [f"M_llm2_run1_gen{gen}_S_doc1000_synt64_lr1e-04_acm_SYNLAST" for gen in range(11)]

labels += [f"M_llm2_run2_gen{gen}_W_doc1000_synt64_lr1e-04_acm_SYNLAST" for gen in range(11)]
labels += [f"M_llm2_run2_gen{gen}_X_doc1000_synt64_lr1e-04_acm_SYNLAST" for gen in range(11)]
labels += [f"M_llm2_run2_gen{gen}_S_doc1000_synt64_lr1e-04_acm_SYNLAST" for gen in range(11)]




device = torch.device("cuda:1")

for m in labels:

    model_name = "dgambettaphd/"+m

    print("hellaswag: " + model_name)

    results = wait_until_model_exists(model_name)

    
    file_path = f"stats_acm/hellaswag/"
    os.makedirs(file_path , exist_ok=True)

    with open(file_path + f'{m}.json', 'w') as fp:
        json.dump(results["results"]["hellaswag"], fp,default=str)

