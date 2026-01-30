# OpenVLA Finetune and Evaluation on Libero Benchmark: A Reproduction Journey

## 🎥 Demos 
- Task 1
Success
(https://github.com/user-attachments/assets/cfc53b0e-e18a-45cd-a768-6cacb226ea37)
Fail
(https://github.com/user-attachments/assets/39667e83-386a-45a6-af3a-17ecc7844d10)
- Task 2
Success

Fail


Loss Curve


## 📖 Project Overview
This project documents the deployment and evaluation of the OpenVLA (7B) model on the Libero-Spatial robot manipulation benchmark. The goal was to validate the model's visual-motor control capabilities in a simulated MuJoCo environment. I have recorded all the related processes in CSDN, [OpenVLA-Learning](https://blog.csdn.net/2303_77547168/article/details/156364335?spm=1011.2415.3001.5331).

## 🛠️ Environment Setup
Successfully running the evaluation required solving several dependency conflicts between legacy `gym` and modern `gymnasium` environments.

**Key Dependencies:**
- Python 3.10
- CUDA 12.x / PyTorch
- `gym < 0.26` (Crucial for Libero compatibility)
- `robosuite` & `libero`
- `openvla`


## 📊 Evaluation Results

| Task Suite | Episodes | Auto-Eval Success Rate | Human-Eval Success Rate |
| :--- | :--- | :--- | :--- |
| Libero Spatial Task1 | 50 | ~14% | ~26%  |
| Libero Spatial Task2 | 8 | ~13% | ~63%  |


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
