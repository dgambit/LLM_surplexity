from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from huggingface_hub import login
import torch

import math
from collections import Counter



def model_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    input_ids = encodings["input_ids"].to(model.device)

    attention_mask = encodings["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Shift dei logits per allinearlia ai target token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    # Calcola le probabilità normalizzate con softmax
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Estrai le probabilità corrispondenti ai token reali
    token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Calcola la loss media (negative log likelihood per token)
    mean_nll = -token_log_probs.mean()

    # Calcola la perplessità
    ppl = torch.exp(mean_nll)
    '''
    token_probs = token_log_probs.exp()
    tokens = tokenizer.convert_ids_to_tokens(shift_labels[0])
    for tok, prob in zip(tokens, token_probs[0].tolist()):
        print(f"{tok}\t{prob:.4f}")
    '''
    return ppl.item()




def model_perplexity_viz(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    input_ids = encodings["input_ids"].to(model.device)

    attention_mask = encodings["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Shift dei logits per allinearlia ai target token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    # Calcola le probabilità normalizzate con softmax
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Estrai le probabilità corrispondenti ai token reali
    token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Calcola la loss media (negative log likelihood per token)
    mean_nll = -token_log_probs.mean()

    # Calcola la perplessità
    ppl = torch.exp(mean_nll)
    
    tok_dict = {}

    token_probs = token_log_probs.exp()
    tokens = tokenizer.convert_ids_to_tokens(shift_labels[0])
    for tok, prob in zip(tokens, token_probs[0].tolist()):
        tok_dict[tok] = prob
    
    return ppl.item(), tok_dict



def first_token_prob(model, tokenizer, document):
    # Tokenizza il documento
    encodings = tokenizer(document, return_tensors="pt")
    input_ids = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  

    last_token_logits = logits[:, -1, :] 
    probs = torch.nn.functional.softmax(last_token_logits, dim=-1)

    max_prob, predicted_token_id = torch.max(probs, dim=-1)

    return max_prob.item()




def text_perplexity(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    freq = Counter(tokens)
        
    probs = {token: count / len(tokens) for token, count in freq.items()}

    H =  -sum(p * math.log2(p) for p in probs.values())/math.log2(len(set(tokens)))
    TPP = 2**H
    return TPP



def text_entropy(tokenizer, text):
    
    tokens = tokenizer.tokenize(text)
    freq = Counter(tokens)
        
    probs = {token: count / len(tokens) for token, count in freq.items()}

    H =  -sum(p * math.log2(p) for p in probs.values())/math.log2(len(set(tokens)))
    
    return H






def NTP(model, tokenizer, prompt, ntop=100):

    encodings = tokenizer(prompt, return_tensors="pt")
    input_ids = encodings["input_ids"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  

    last_token_logits = logits[:, -1, :] 
    probs = torch.nn.functional.softmax(last_token_logits, dim=-1)

    top_probs, top_token_ids = torch.topk(probs, ntop, dim=-1)

    top_tokens = [tokenizer.decode([token_id]) for token_id in top_token_ids[0]]
    top_probs = top_probs[0].tolist()

    return {"tokens": top_tokens, "probs": top_probs}




import pandas as pd
from datasets import load_dataset

import spacy

import os

import textdescriptives as td
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("textdescriptives/all") 


import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')


import re
import math
import pickle


from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def cleantext(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = " ".join(filtered_words)
    return filtered_text


def TTR(text):
    text = cleantext(text.lower())
    text = re.sub(r'[^a-zA-Z ]', "", text)
    token = text.split(" ")
    token = [t for t in token if (t!="" and t not in stop_words)]
    type = set(token)
    if len(token) !=0:
        TTR = len(type)/len(token)
    else:
        TTR=0
    return TTR


def lang_entropy(text, base=2):
    text = cleantext(text.lower())
    text = re.sub(r'[^a-zA-Z ]', "", text)
    token = text.split(" ")
    token = [t for t in token if (t!="" and t not in stop_words)]
    type = set(token)
    occurs = {t:0 for t in type}
    for t in token:
        occurs[t] +=1
    probs = occurs.values()
    probs = [p/len(token) for p in probs]
    H = -sum([p  * math.log(p) / math.log(base) for p in probs ])
    try:
        H = H/(-sum([1/len(token) * math.log(1/len(token)) / math.log(base) for t in token ]))
    except:
        H = 0
    return H


from collections import Counter

def count_words(word_list):
    return Counter(word_list)


nltk.download('punkt')

def process_text_tokens(text):
    text = str(text)
    stemmer = PorterStemmer()
    text = text.lower()    
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    processed_text = ' '.join(stemmed_words)
    return processed_text


def count_tokens(sent):
    doc = nlp(sent)
    token_dict = {}
    for token in doc:
        try: token_dict[token.pos_].append(process_text_tokens(token))
        except: token_dict[token.pos_] = [process_text_tokens(token)]
    #token_dict["ALL"] = sum(token_dict.values())
    return token_dict



def stats(x):
    doc = nlp(x)
    return doc._.descriptive_stats | doc._.information_theory


def avg(list):
    return(sum(list)/len(list))
