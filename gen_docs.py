

from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from huggingface_hub import login
import random
import time
import metrics

from datasets import concatenate_datasets, load_dataset



model_name_dict = {"llama-3-8b": "llm3",
                "llama-2-7b": "llm2",
                "gemma-3n-E4B": "gmm3",
                "mistral-7b": "mis", 
                "DeepSeek-V3": "ds3",
                "gemma-2-2b":"gmm2"}


def gen_docs(gen, run, modelname, datasetname, ndoc, real, synt,rnd, add_name):
    hf_token = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    login(token = hf_token) #writeToken

    
    #curr_model = f"dgambettaphd/M_gen{gen}_run{run}_{modelname}_{datasetname}_doc{ndoc}_real{real}_synt{synt}"+add_name
    
    curr_model = f"dgambettaphd/M_{model_name_dict[modelname]}_run{run}_gen{gen}_{datasetname}_doc{ndoc}_synt{synt}"+add_name

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = curr_model,
        max_seq_length = 512,
        dtype = None,
        load_in_4bit = True,
        device_map = {"": 0})

    
    FastLanguageModel.for_inference(model)

    def truncate_text_by_tokens(example, tokenizer, max_tokens):
            input_ids = tokenizer(example["text"], return_tensors="pt")["input_ids"]
            truncated_ids = input_ids[0][:max_tokens]
            truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
            example["text"] = truncated_text
            return example
    

    if gen > 0:
        web_url = f"dgambettaphd/D_{model_name_dict[modelname]}_run{run}_gen{gen-1}_{datasetname}_doc{ndoc}_synt{synt}"+add_name
        web = load_dataset(web_url, split="train")
        columns_to_select = ["id_doc", 'text', 'dataset', "gen", "synt", "lang_entropy"]  # Le colonne che desideri mantenere
        web = web.select_columns(columns_to_select)

    else:
        web_url = "dgambettaphd/prompt_wxs_5000doc"
        web = load_dataset(web_url, split="train")
        columns_to_select = ["id_doc", 'text', 'dataset', "gen", "synt"]  # Le colonne che desideri mantenere
        web = web.select_columns(columns_to_select)
        web = web.map(truncate_text_by_tokens,fn_kwargs={"tokenizer": tokenizer, "max_tokens": real + synt})
        web = web.add_column('lang_entropy', [metrics.lang_entropy(text) for text in web["text"]])

        


    prompt_url = f"dgambettaphd/prompt_wxs_5000doc"

    prompts = load_dataset(prompt_url, split="train")
    prompts = prompts.filter(lambda example: example['dataset'] in datasetname)
    

    
    random.seed(run*10 + gen)
    
    
    ids_prompt = random.sample(range(len(prompts)), ndoc)



    dict_docs = {"id_doc": [],
                "text": [],
                "dataset": [],
                "gen":[],
                "synt": [],
                "lang_entropy": []}



    device = model.device  # Ottieni il dispositivo del modello

    
    for i in range(ndoc):
        
        prompt = prompts["text"][ids_prompt[i]]
        ds = prompts["dataset"][ids_prompt[i]]
        id_doc = prompts["id_doc"][ids_prompt[i]]
        input_tot = tokenizer.encode(prompt, return_tensors="pt").to(device)
        input = input_tot[:, :real] 

        if synt>0:
            output = model.generate(input, max_new_tokens= synt, min_new_tokens=synt-8)
            doc = tokenizer.decode(output[0])
        else:
            doc = tokenizer.decode(input[0])
        
    
        doc = doc.replace("<s>", "")
        doc = doc.replace("<|begin_of_text|>","")
        doc = doc.replace("<|end_of_text|>", "")
        
    
        
        if i%100 == 0:
            print(i)
            print(doc)

        dict_docs["text"].append(doc)
        dict_docs["id_doc"].append(id_doc)
        dict_docs["synt"].append(synt)
        dict_docs["dataset"].append(ds)
        dict_docs["gen"].append(gen)
        dict_docs["lang_entropy"].append(metrics.lang_entropy(doc))
        #web = web.add_column('lang_entropy', [metrics.lang_entropy(text) for text in web["text"]])



    new_dataset = Dataset.from_dict(dict_docs)

    update_dataset = concatenate_datasets([web,new_dataset ])
    
    update_dataset = update_dataset.add_column('MPP', [metrics.model_perplexity(model, tokenizer, text) for text in update_dataset["text"]])

    update_dataset.push_to_hub(f"D_{model_name_dict[modelname]}_run{run}_gen{gen}_{datasetname}_doc{ndoc}_synt{synt}"+add_name)



