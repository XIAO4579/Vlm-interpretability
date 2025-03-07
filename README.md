# Towards Understanding How Knowledge Evolves in Large Vision-Language Models

Code for the CVPR 2024 paper "Towards Understanding How Knowledge Evolves in Large Vision-Language Models"

## Overview!
![distribution](https://github.com/user-attachments/assets/2a9cdc03-18a5-4047-81c2-6d4f2fe53b41)

Large Vision-Language Models (LVLMs) are gradually becoming the foundation for many artificial intelligence applications. However, understanding their internal working mechanisms has continued to puzzle researchers, which in turn limits the further enhancement of their capabilities. In this paper, we seek to investigate how multimodal knowledge evolves and eventually induces natural languages in LVLMs. We design a series of novel strategies for analyzing internal knowledge within LVLMs, and delve into the evolution of multimodal knowledge from three levels, including single token probabilities, token probability distributions, and feature encodings. In this process, we identify two key nodes in knowledge evolution: the critical layers and the mutation layers, dividing the evolution process into three stages: rapid evolution, stabilization, and mutation. Our research is the first to reveal the trajectory of knowledge evolution in LVLMs, providing a fresh perspective for understanding their underlying mechanisms.

## Setup
The environment should match the model you intend to analyze. For instance, if you are using LLaVA-1.5, please set up your environment following the official guidelines in the LLaVA GitHub repository: https://github.com/haotian-liu/LLaVA. 

## Experiments
For every open-source vlm model, please add
```
output_hidden_states=True
return_dict_in_generate=True
```
in the generate function. See more details in Jupyter Notebook codes.

To get the heatmap image, run [```heatmap.ipynb```](heatmap.ipynb)

## Citation
If you find our project interesting, we hope you can star our repo and cite our paper as follows:
