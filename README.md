Code for the paper "[Effective Distillation of Table-based Reasoning Ability from LLMs](https://aclanthology.org/2024.lrec-main.492/)"

## Dataset
We enriched the [SciGen](https://github.com/UKPLab/SciGen) dataset
````
├── cot
       └── `train.source`    
       └── `train.target`       
       └── `val.source` 
       └── `val.target` 
       └── `test.source` 
       └── `test.target`
````
source include the table and generated reasoning and target includes the generated description.
## Generating Table Reasoning
````
python promptDiverse.py
````

## Finetuning
````
bash /flan-t5_train.sh
````

## Evaluating

````
bash flan-run-tapex.sh
````
## Citation
Please use the following citation:

````
@inproceedings{yang-etal-2024-effective,
    title = "Effective Distillation of Table-based Reasoning Ability from {LLM}s",
    author = "Yang, Bohao  and
      Tang, Chen  and
      Zhao, Kun  and
      Xiao, Chenghao  and
      Lin, Chenghua",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.492",
    pages = "5538--5550",
    abstract = "Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing tasks. However, their enormous parameter size and extremely high requirements for compute power pose challenges for their practical deployment. Recent research has revealed that specific capabilities of LLMs, such as numerical reasoning, can be transferred to smaller models through distillation. Some studies explore the potential of leveraging LLMs to perform table-based reasoning. However, there has been no prior work focusing on table reasoning skills in smaller models specifically tailored for scientific table-to-text generation tasks. In this paper, we propose a novel table-based reasoning distillation approach, with the aim of distilling LLMs into tailored smaller models. Our experimental results have shown that a 220 million parameter model (Flan-T5-base) fine-tuned using distilled data, not only achieves a significant improvement compared to traditionally fine-tuned baselines, but also surpasses specific LLMs on a scientific table-to-text generation dataset. Our code is available at https://github.com/Bernard-Yang/DistillTableCoT.",
}
````

````
@article{moosavi:2021:SciGen,
  author    = {Nafise Sadat Moosavi, Andreas R{\"u}ckl{\'e}, Dan Roth, Iryna Gurevych},
  title     = {Learning to Reason for Text Generation from Scientific Tables},
  journal   = {arXiv preprint arXiv:2104.08296},
  year      = {2021},
  url       = {https://arxiv.org/abs/2104.08296}
}
````
