# OpenVLA Finetune and Evaluation on Libero Spatial Benchmark

## 🎥 Demos 
- Task 1
Success

https://github.com/user-attachments/assets/5a1b6eab-6c3e-48ca-a834-d62189140e08

Fail

https://github.com/user-attachments/assets/39667e83-386a-45a6-af3a-17ecc7844d10

- Task 2
Success

https://github.com/user-attachments/assets/9652c4bc-4b5a-4d14-8973-5ded73d3c259

Fail

https://github.com/user-attachments/assets/717102b1-5762-4e76-bf8a-a8bdb012169a

## Loss Curve

<img width="2840" height="1418" alt="image" src="https://github.com/user-attachments/assets/5ebf3339-bd7a-4795-812e-f2e7245ecf75" />

The curve shows an “L-shaped” pattern. It drops rapidly from around 12 at the beginning, and then decreases slowly and steadily to the 2–3 range. The whole process can be divided into:

1. **Rapid decline phase:** indicating the model quickly learned the basic rules of the task (e.g., the rough correspondence between images and actions);
2. **Gradual decline phase:** the slower decrease later suggests the model is “fine-tuning,” learning more subtle operational details;
3. **Oscillations:** the small sawtooth-like fluctuations in the curve are completely normal (since the difficulty varies across batches). As long as the overall trend keeps going downward, the training is healthy.

## 📖 Project Overview
This project accomplishes the Fine-tuning and evaluation of the OpenVLA (7B) model on the Libero-Spatial robot manipulation benchmark. I re-normalized the final action outputs of the VLA model to ensure the actions are within a reasonable range; originally, the VLA’s predicted action magnitudes were so small that it could not complete the task.

## 🛠️ Environment Setup
- OS: Ubuntu 22.04
- Python: 3.10
- CUDA: 12.1
- Torch: 2.2.0
- GPU: RTX 4090
- OpenVLA base


## 📊 Evaluation Results

My training configuration is as follows: batch size = 1, gradient accumulation steps = 16, and LoRA rank = 16. The training speed is about 4 s/iteration with approximately 18 GB of GPU memory usage. Fine-tuning for 5,000 steps takes around 6 hours.

| Task Suite | Episodes | Success Rate |
| :--- | :--- | :--- |
| Libero Spatial Task1 | 50 | 10% |
| Libero Spatial Task2 | 50 | 20% |


## Acknowledgements 
This project is based on the OpenVLA codebase. Special thanks to the original authors for their open-source contribution.



---

#### Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2406.09246):

```bibtex
@article{kim24openvla,
    title={OpenVLA: An Open-Source Vision-Language-Action Model},
    author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
    journal = {arXiv preprint arXiv:2406.09246},
    year={2024}
} 
```
