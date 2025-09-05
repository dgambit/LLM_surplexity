from huggingface_hub import login
from unsloth import FastLanguageModel
from huggingface_hub import login, HfApi


import fine_tuning
import gen_docs
import gen_model_zero


def pipeline(modelname = "llama-3-8b", datasetname="WXS", start=-1, ndoc=1000, run =0, real=64, synt=64,lr = 2e-4, num_train_epochs=5,rnd=42, add_name="", filter = ""):
        if filter != "":
                add_name += "_"+ filter

        if start == -1:
                gen_model_zero.gen_model_zero(run, modelname, datasetname, ndoc, real, synt,rnd, add_name)
                gen_docs.gen_docs(0, run, modelname, datasetname, ndoc, real, synt,rnd, add_name) 
                start = 0 

        for gen in range(start, 10):

                print(gen, run, modelname, datasetname, ndoc,  real, synt,rnd, add_name)

                fine_tuning.fine_tuning(gen, run, modelname, datasetname, ndoc, real, synt , lr, num_train_epochs,rnd, add_name, filter) 
                gen_docs.gen_docs(gen+1, run, modelname, datasetname, ndoc, real, synt,rnd, add_name) 


