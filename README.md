Here’s a polished and cohesive **README.md** file:

```markdown
# MedGAN: Medical Image Generation with GANs

This repository provides implementations of various Generative Adversarial Networks (GANs) for generating medical images, with a focus on brain tumor MRI scans. It includes popular GAN architectures optimized for high-quality and diverse image generation.

---

## Implemented GAN Architectures

The following GAN architectures are implemented in this project:

- **DCGAN (Deep Convolutional GAN):**  
  [Paper](https://arxiv.org/abs/1511.06434)  
  Employs convolutional layers to create high-quality image outputs.

- **ProGAN (Progressive Growing of GANs):**  
  [Paper](https://arxiv.org/abs/1710.10196)  
  Uses progressive training to improve stability, quality, and variation in generated images.

- **StyleGAN & StyleGAN2:**  
  - **StyleGAN:** [Paper](https://arxiv.org/abs/1812.04948)  
    A style-based generator offering granular control over image features.  
  - **StyleGAN2:** [Paper](https://arxiv.org/abs/1912.04958)  
    Builds on StyleGAN with better image quality and reduced artifacts.

- **WGAN (Wasserstein GAN):**  
  [Paper](https://arxiv.org/abs/1701.07875)  
  Improves training stability by minimizing Wasserstein distance.

- **CycleGAN:**  
  [Paper](https://arxiv.org/abs/1703.10593)  
  Focuses on image-to-image translation with cycle consistency.

---

## Dataset

The project utilizes the [Brain Tumor MRI Scans dataset](https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans?select=glioma), an open-source dataset available on Kaggle.  
Ensure you download and place it in the `dataset` directory before training.

---

## Getting Started

Follow these steps to set up the project and train the models:

### 1. Clone the Repository
```bash
git clone https://github.com/mozaloom/medgan
cd medgan
```

### 2. Install Dependencies
Install all required Python packages using:
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Download the [Brain Tumor MRI Scans dataset](https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans?select=glioma) and place it in the `dataset` directory.

### 4. Train a Model
Each GAN has its own training script. Navigate to the specific GAN folder (e.g., `ProGAN/`, `StyleGAN2/`) and refer to the README or script for detailed instructions.

---

## Examples of Generated Images

Below is an example image generated using **ProGAN** after 180 epochs (Step 5):

![Generated Image](ProGan/ProGan-180-Epochs-5-Steps/step5/img_50.png)

For more examples, visit the [images directory](images).

---

## Contributing

Contributions are welcome! To improve this project:

1. Fork the repository.
2. Create a new branch (`feature/my-feature`).
3. Commit your changes.
4. Open a pull request.

Feel free to report issues or suggest enhancements.

---

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

## Contact

For any inquiries, suggestions, or collaboration opportunities, please feel free to reach out.
```

### Key Improvements:
- Organized sections for clarity.
- Added direct dataset and GAN-specific script instructions.
- Unified the tone and formatting.  
This should now function as a complete and professional `README.md`.
