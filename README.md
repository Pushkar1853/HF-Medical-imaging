# Hugging Face 🤗 - Medical Imaging models
Repository for vision model 

## SAM (Segment-Anything Model):
- performing inference with MedSAM: `segment-anything/run-inference-with-medsam.ipynb`
- fine-tuning **SamModel** on a custom dataset: ` segment-anything/fine-tune-sam-on-custom-dataset.ipynb`
- official paper: [Segment Anything](https://github.com/Pushkar1853/HF-Medical-imaging/blob/14decd3bce0a65219cb9fa5c3230a0a0e4860fe2/segment-anything/SAM-paper.pdf)
- model scripts available here: ` segment-anything/modeling `
- util functions scripts available here: ` segment-anything/utils `
- see predictor example: ` segment-anything/predictor_example.ipynb `

## DINO V2:
- finetuning **DINOv2** on a custom dataset: ` DINOV2/dinov2_finetune.py `
- run this Python file through the steps mentioned in ` DINOV2/tester-dinov2.ipynb `
- performing inference: ` DINOV2/visualize-self-attention-of-dino.ipynb `
- official paper: [DINOv2: Learning Robust Visual Features without Supervision](https://github.com/Pushkar1853/HF-Medical-imaging/blob/14decd3bce0a65219cb9fa5c3230a0a0e4860fe2/DINOV2/DINOV2-paper.pdf)
  
