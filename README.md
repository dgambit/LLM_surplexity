# Learning by Surprise: Surplexity for Mitigating Model Collapse in Generative AI
## Table of contents
1. [Citing](#citing)
2. [Packages](#packages)
3. [Abstract](#abstract)

# Citing
In this repository you can find the code for running our simulation framework and to replicate the analysis conducted in our paper.
If you use the code in this repository, please cite our paper:

* Daniele Gambetta, Gizem Gezici, Fosca Giannotti, Dino Pedreschi, Alistair Knott, Luca Pappalardo. "Learning by Surprise: Surplexity for Mitigating Model Collapse in Generative AI" 
arXiv preprint arXiv:2410.12341*

```
@misc{gambetta2025learningsurprisesurplexitymitigating,
      title={Learning by Surprise: Surplexity for Mitigating Model Collapse in Generative AI}, 
      author={Daniele Gambetta and Gizem Gezici and Fosca Giannotti and Dino Pedreschi and Alistair Knott and Luca Pappalardo},
      year={2025},
      eprint={2410.12341},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.12341}, 
}
```
# Packages
For running notebooks and scripts of this project you must install the following Python packages:
```
  pandas
  matplotlib
  numpy
  huggingface_hub
  unsloth
  transformers
```

# Abstract
As synthetic content increasingly infiltrates the web, generative AI models may be retrained on their own outputs: a process termed "autophagy". This leads to model collapse: a progressive loss of performance and diversity across generations. Recent studies have examined the emergence of model collapse across various generative AI models and data types, and have proposed mitigation strategies that rely on incorporating human-authored content. However, current characterizations of model collapse remain limited, and existing mitigation methods assume reliable knowledge of whether training data is human-authored or AI-generated. In this paper, we address these gaps by introducing new measures that characterise collapse directly from a model's next-token probability distributions, rather than from properties of AI-generated text. Using these measures, we show that the degree of collapse depends on the complexity of the initial training set, as well as on the extent of autophagy. Our experiments prompt a new suggestion: that model collapse occurs when a model trains on data that does not "surprise" it. We express this hypothesis in terms of the well-known Free Energy Principle in cognitive science. Building on this insight, we propose a practical mitigation strategy: filtering training items by high surplexity, maximising the surprise of the model. Unlike existing methods, this approach does not require distinguishing between human- and AI-generated data. Experiments across datasets and models demonstrate that our strategy is at least as effective as human-data baselines, and even more effective in reducing distributional skewedness. Our results provide a richer understanding of model collapse and point toward more resilient approaches for training generative AI systems in environments increasingly saturated with synthetic data. 


