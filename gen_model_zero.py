

from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from huggingface_hub import login
from huggingface_hub import login, HfApi


model_name_dict = {"llama-3-8b": "llm3",
                "llama-2-7b": "llm2",
                "gemma-3n-E4B": "gmm3",
                "mistral-7b": "mis", 
                "DeepSeek-V3": "ds3",
                "gemma-2-2b":"gmm2"}


def gen_model_zero(run, modelname, datasetname, ndoc, real, synt,rnd, add_name):

    base_model = f"unsloth/{modelname}"

    hf_token = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    login(token = hf_token) 

    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = base_model,
            max_seq_length = 512,
            dtype = None,
            load_in_4bit = True,
            device_map = {"": 0})

    #FastLanguageModel.for_inference(model)


    zero_model = f"dgambettaphd/M_{model_name_dict[modelname]}_run{run}_gen0_{datasetname}_doc{ndoc}_synt{synt}"+add_name

    repo_name = zero_model
    api = HfApi()
    api.create_repo(repo_name, private=False)

    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)    
