

from unsloth import FastLanguageModel

import os
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
from unsloth import unsloth_train


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

from huggingface_hub import login, HfApi, create_repo, Repository

import random


model_name_dict = {"llama-3-8b": "llm3",
                "llama-2-7b": "llm2",
                "gemma-3n-E4B": "gmm3",
                "mistral-7b": "mis", 
                "DeepSeek-V3": "ds3",
                "gemma-2-2b":"gmm2"}


def fine_tuning(gen, run, modelname, datasetname, ndoc, real, synt,lr, num_train_epochs,rnd, add_name, filter):

    
    print(f"Fine tuning of gen {gen} run {run}")
 
    #curr_model = f"dgambettaphd/M_gen{gen}_run{run}_{modelname}_{datasetname}_doc{ndoc}_real{real}_synt{synt}"+add_name
    #curr_doc = f"dgambettaphd/D_gen{gen}_run{run}_{modelname}_{datasetname}_doc{ndoc}_real{real}_synt{synt}"+add_name

    curr_model = f"dgambettaphd/M_{model_name_dict[modelname]}_run{run}_gen{gen}_{datasetname}_doc{ndoc}_synt{synt}"+add_name
    curr_doc = f"dgambettaphd/D_{model_name_dict[modelname]}_run{run}_gen{gen}_{datasetname}_doc{ndoc}_synt{synt}"+add_name


    hf_token = "xxxxxxxxxxxxxxxxxxxxxxxxxxx"
    login(token = hf_token) #writeToken

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = curr_model, 
        max_seq_length = 2048,
        dtype = None,
       load_in_4bit = True,
        device_map = {"": 0}    )


    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        return { "text" : [example + EOS_TOKEN for example in examples["text"]] }
    
    dataset = load_dataset(curr_doc, split = "train")
    

    if filter == "SYNALL":
        dataset = dataset.filter(lambda example: example["synt"] > 0) 
        dataset = dataset.shuffle(seed=42).select(range(ndoc))

    if filter == "SYNLAST":
        dataset = dataset.filter(lambda example: example["gen"] == gen)
        dataset = dataset.shuffle(seed=42).select(range(ndoc))

    if filter == "FRESH":
        dataset = dataset.filter(lambda example: example["synt"] == 0) 
        
        dataset = dataset.shuffle(seed=run*10 + gen).select(range(ndoc))
        
    if filter == "MPP":
        dataset = dataset.sort("MPP", reverse= True)
        dataset = dataset.select(range(ndoc))

    if filter == "LANG":
        dataset = dataset.sort("lang_entropy", reverse= True)
        dataset = dataset.select(range(ndoc))

    if filter == "TPP":
        dataset = dataset.sort("TPP", reverse= True)
        dataset = dataset.select(range(ndoc))

    if filter == "TFP":
        dataset = dataset.sort("TFP", reverse= False)
        dataset = dataset.select(range(ndoc))

    if filter == "RANDOM":
        dataset = dataset.shuffle(seed=42).select(range(ndoc))

    
    dataset = dataset.map(formatting_prompts_func, batched = True,)


    


    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 42,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    


    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 8,

        args = UnslothTrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 8,
        warmup_steps = 5,
        num_train_epochs = num_train_epochs, # Set this for 1 full training run.
        max_steps = -1,
        learning_rate = lr,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
        report_to = "none",
        ),
    )

    
    trainer_stats = trainer.train()

    directory = "./log_finetuning"
    os.makedirs(directory, exist_ok=True)

    
    #file_path = f"{directory}/log_gen{gen}_run{run}_{modelname}_{datasetname}_doc{ndoc}_real{real}_synt{synt}{add_name}.txt"


    #with open(file_path, "w") as file:
    #    file.write(str(trainer_stats))



    next_model = f"M_{model_name_dict[modelname]}_run{run}_gen{gen+1}_{datasetname}_doc{ndoc}_synt{synt}"+add_name

    repo_name = next_model
    api = HfApi()
    api.create_repo(repo_name, private=False)

    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)    


