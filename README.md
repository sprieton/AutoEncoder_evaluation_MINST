# 🧠 Deep Autoencoder Implementation on MNIST and Fashion-MNIST

## 📘 Project Overview
This project explores the implementation and evaluation of **deep autoencoders** built with **dense neural networks** to perform dimensionality reduction and denoising on the **MNIST** and **Fashion-MNIST (FMNIST)** datasets.  
The objective is to compare different network architectures and hyperparameters, analyze their reconstruction quality, and extend the best-performing model into a **denoising autoencoder**.

The work was developed in **Python** using **Jupyter Notebook**, as part of a laboratory exercise focused on deep learning model design and evaluation.

---

## 🧩 Objectives
1. **Implement** autoencoders with multiple fully connected layers (dense networks).
2. **Compare architectures** with 3 and 5 layers for both encoder and decoder parts.
3. **Study the effect** of different latent dimensions (`15`, `30`, `50`, `100`).
4. **Introduce regularization techniques**, including:
   - Weight regularization (e.g., L2).
   - Output regularization via **Lasso penalty** (L1-norm of encoder output |E|).
5. **Evaluate performance** using:
   - **Peak Signal-to-Noise Ratio (PSNR)** as a quantitative reconstruction metric.
   - Visual comparisons between original and reconstructed/denoised images.
6. **Develop and analyze** a **denoising autoencoder**, injecting zero-mean Gaussian noise with tunable variance and studying its impact on reconstruction performance.

---

## ⚙️ Methodology Summary
1. **Data Preparation**  
   Both MNIST and FMNIST datasets were normalized and reshaped for input into dense layers.

2. **Model Architectures**  
   Two families of deep autoencoders were tested:
   - **3-layer** encoder/decoder
   - **5-layer** encoder/decoder  
   Each was trained with varying latent dimensions to evaluate the trade-off between compression and reconstruction fidelity.

3. **Regularization**  
   To improve generalization and stability, regularization techniques were applied, including:
   - Weight decay (L2 regularization)
   - L1 penalty on the encoder’s latent representation

4. **Evaluation Metrics**  
   Reconstruction quality was assessed through **PSNR**, providing an interpretable signal-quality measure.

5. **Denoising Autoencoder**  
   For the best-performing architecture, Gaussian noise was added to input images.  
   The model’s ability to reconstruct clean images from noisy inputs was analyzed across different noise variances.

---

## 📊 Results Overview
- **Architectural comparison** showed that deeper models can yield better reconstructions at the cost of increased training time.  
- **Latent dimension size** significantly affected performance, with mid-range dimensions (≈30–50) often achieving a balance between compression and quality.  
- **Regularization** improved robustness and prevented overfitting.  
- The **denoising autoencoder** effectively removed Gaussian noise, maintaining high PSNR even at moderate noise levels.

---

## 💻 Tools and Dependencies
The project was implemented in Python using:
- `TensorFlow / Keras`
- `NumPy`
- `Matplotlib`
- `scikit-learn`

Make sure to install the dependencies before running the notebook:
```bash
pip install tensorflow numpy matplotlib scikit-learn
