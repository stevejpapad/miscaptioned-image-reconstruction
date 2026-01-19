# Latent Reconstruction from Generated Data for Multimodal Misinformation Detection

Repository for the paper "Latent Reconstruction from Generated Data for Multimodal Misinformation Detection". You can read the pre-print here: https://arxiv.org/abs/2504.06010

## Abstract
>*Multimodal misinformation, such as miscaptioned images, where captions misrepresent an image's origin, context, or meaning, poses a growing challenge in the digital age. Due to the scarcity of large-scale annotated datasets for multimodal misinformation detection (MMD), recent approaches rely on synthetic training data created via out-of-context pairings or named entity manipulations (e.g., altering names, dates, or locations). However, these often yield simplistic, unrealistic examples, which limits their utility as training examples. To address this, we introduce "MisCaption This!", a framework for generating high-fidelity synthetic miscaptioned datasets through Adversarial Prompting of Vision-Language Models (VLMs). Additionally, we introduce "Latent Multimodal Reconstruction" (LAMAR), a Transformer-based network trained to reconstruct the embeddings of truthful captions, providing a strong auxiliary signal to guide detection. We explore various training strategies (end-to-end vs. large-scale pre-training) and integration mechanisms (direct, mask, gate, and attention). Extensive experiments show that models trained on "MisCaption This!" data generalize better to real-world misinformation, while LAMAR achieves new state-of-the-art on NewsCLIPpings, VERITE, and the newly introduced VERITE 24/25 benchmark; highlighting the efficacy of VLM-generated data and reconstruction-based networks for advancing MMD.*

![Screenshot](docs/pipeline.png)

## Preparation
- Clone this repo: 
```
git clone https://github.com/stevejpapad/miscaptioned-image-reconstruction/
cd miscaptioned-image-reconstruction/
```
- Create a python (>= 3.9) environment (Anaconda is recommended) 
- Install all dependencies with: `pip install -r requirements.txt`.

## Datasets
Access to the "Miscaption This!" dataset is available upon request and is intended solely for research purposes.
If you want to reproduce the experiments of the paper, it is necessary to first download the following datasets and save them in their respective folder: 
- VisualNews -> https://github.com/FuxiaoLiu/VisualNews-Repository -> `VisualNews/`
- NewsCLIPings -> https://github.com/g-luo/news_clippings -> `news_clippings/`
- VERITE -> https://github.com/stevejpapad/image-text-verification -> `VERITE/`

If you encounter any problems while downloading and preparing VERITE (e.g., broken image URLs), please contact stefpapad@iti.gr

## Reproducibility
To extract CLIP features and reproduce all experiments run: 
```python src/main.py``` 

## Citation 
If you find our work useful, please cite:
```
@article{papadopoulos2025latent,
      title={Latent Multimodal Reconstruction for Misinformation Detection}, 
      author={Papadopoulos, Stefanos-Iordanis and Koutlis, Christos and Papadopoulos, Symeon and Petrantonakis, Panagiotis C},
      journal={arXiv preprint arXiv:2504.06010},      
      year={2025}
}
```

## Acknowledgements
This work is partially funded by the projects: "vera.ai: VERification Assisted by Artificial Intelligence" under grant agreement no. 101070093, "DisAI - Improving scientific excellence and creativity in combating disinformation with artificial intelligence and language technologies" under grant agreement no. 101079164, and  "AI4TRUST
AI-based-technologies for trustworthy solutions against disinformation" under grant agreement no. 101070190.

## Licence
This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/stevejpapad/miscaptioned-image-reconstruction/blob/main/LICENSE) file for more details.

## Contact
Stefanos-Iordanis Papadopoulos (stefpapad@iti.gr)

