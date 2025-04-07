# Towards Understanding How Knowledge Evolves in Large Vision-Language Models

Code for the CVPR 2025 paper "Towards Understanding How Knowledge Evolves in Large Vision-Language Models"

## Overview
![path](https://github.com/user-attachments/assets/09969d8e-2698-4f22-8a24-05be6e90b32b)
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

To get the heatmap image, please run [```heatmap.py```](heatmap.py)
<img src="https://github.com/user-attachments/assets/2a9cdc03-18a5-4047-81c2-6d4f2fe53b41" alt="Amber25_0" width="1000" height="400">

To get the token line chart image, please run [```plot_token_probabilities_area.py```](plot_token_probabilities_area.py)
<img src="https://github.com/user-attachments/assets/1f190764-6eac-405a-8653-c68622ae2289" alt="Amber25_0" width="800">

To get the simple tsne image, please run [```tsne.py```](tsne.py)
<img src="https://github.com/user-attachments/assets/c28fe404-3d65-4e3c-a4dc-51d3f681f521" alt="Amber25_0" width="800" height="550">


To get the combined tsne image, please run [```combined_tsne.py```](combined_tsne.py)
<img src="https://github.com/user-attachments/assets/bdd98ab1-67b4-4128-ba1f-39bda6474e8d" alt="Amber25_0" width="800" height="550">

## Citation
If you find our project interesting, we hope you can star our repo and cite our paper as follows:
```
@misc{wang2025understandingknowledgeevolveslarge,
      title={Towards Understanding How Knowledge Evolves in Large Vision-Language Models}, 
      author={Sudong Wang and Yunjian Zhang and Yao Zhu and Jianing Li and Zizhe Wang and Yanwei Liu and Xiangyang Ji},
      year={2025},
      eprint={2504.02862},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.02862}, 
}
```
