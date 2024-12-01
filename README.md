
```markdown
# medgan

This repository contains implementations of various Generative Adversarial Networks (GANs) for medical image generation.

## Implemented GAN Architectures


* **DCGANs:** 
    * [Paper](https://arxiv.org/abs/1511.06434)
    * Deep Convolutional GANs for generating high-quality images.
* **ProGAN:** 
    * [Paper](https://arxiv.org/abs/1710.10196)
    * Progressive Growing of GANs for Improved Quality, Stability, and Variation.
* **StyleGAN1 / StyleGAN-30-Epoch-5-Steps:** 
    * [Paper](https://arxiv.org/abs/1812.04948)
    * A style-based generator architecture for GANs.
* **StyleGAN2:** 
    * [Paper](https://arxiv.org/abs/1912.04958)
    * Improved StyleGAN with better image quality and fewer artifacts.
* **WGAN:** 
    * [Paper](https://arxiv.org/abs/1701.07875)
    * Wasserstein GAN with improved training stability.
* **CycleGAN:**  
    * [Paper](https://arxiv.org/abs/1703.10593)
    * Image-to-image translation using cycle-consistent adversarial networks.

## Dataset

This project uses an open-source dataset of brain tumor MRI scans from Kaggle:

* [Brain Tumor MRI Scans](https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans?select=glioma)

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mozaloom/medgan
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   Download the dataset from the link above and place it in the `dataset` directory.

4. **Train a model:**
   Each GAN implementation has its own training script. Refer to the individual folders for specific instructions.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests.

## License

This project is licensed under the [LICENSE](LICENSE) file.
```


## Examples of Generated Images

* **[View generated images here](images)** 
    *  **Example from ProGAN (after 180 epochs, step 5):**

      ![Generated Image](ProGan/ProGan-180-Epochs-5-Steps/step5/img_50.png)


## License

This project is licensed under the [LICENSE](LICENSE) file.

```
```
