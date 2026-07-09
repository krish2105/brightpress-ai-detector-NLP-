<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <h1>🤖 BrightPress AI Text Detector</h1>
  <p><i>A GAN-Style Adversarial NLP Build: Generator vs Discriminator</i></p>
</div>

---

## 📖 Overview
The **BrightPress AI Text Detector** is an advanced NLP application built to distinguish between human-written text and AI-generated content. Developed as a case study for BrightPress (a media and online-education company), this project employs an adversarial (GAN-style) training loop where a Generator (`gpt2`) attempts to fool a Discriminator (`roberta-base`).

The end result is a highly accurate classification model deployed via a stunning, responsive **Streamlit** dashboard.

## ✨ Features
- ⚔️ **Adversarial Training Pipeline:** Uses a generative model to dynamically create synthetic data, forcing the discriminator to learn deep linguistic patterns rather than simple heuristics.
- 🎨 **Modern Streamlit Dashboard:** Features a beautiful UI with custom CSS, glassmorphism elements, and fully responsive Dark/Light modes.
- 🔍 **Single Scan Mode:** Paste an essay and receive an instant probability score (Human vs. AI) via an interactive gauge chart.
- 🎭 **Face-Off Mode:** Compare a known human text side-by-side with suspected AI text to see how the model evaluates them simultaneously.
- ☁️ **Cloud Ready:** Fully configured with `requirements.txt` for one-click deployment to Streamlit Community Cloud.

## 🛠️ Architecture
- **Generator:** `gpt2` (Generates synthetic AI texts using randomized temperature scaling).
- **Discriminator:** `roberta-base` (Fine-tuned on 2,000 texts with learning rate warmup and weight decay).
- **Frontend UI:** Streamlit with custom injected HTML/CSS.
- **Data Source:** IMDb dataset for high-quality human text baselines.

## 🚀 Getting Started

### Local Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/krish2105/brightpress-ai-detector-NLP-.git
   cd brightpress-ai-detector-NLP-
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit Dashboard:
   ```bash
   streamlit run app.py
   ```

### 🧠 Model Training (Google Colab)
To achieve maximum accuracy, you must train the model using the provided Jupyter Notebook:
1. Upload `AI_Text_Detector_GAN_Adversarial_Loop.ipynb` to Google Colab.
2. Ensure you are connected to a GPU instance (`Runtime > Change runtime type > T4 GPU`).
3. Click `Runtime > Run All`. 
4. Once training completes, download the generated `detector` folder and place it in your local repository folder.

## 🌐 Cloud Deployment
This project is configured for instant deployment on Streamlit Community Cloud:
1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Connect your GitHub account and select this repository.
3. Set the main file path to `app.py`.
4. Click **Deploy**.
*(Note: If the `detector` folder is absent during deployment, the app automatically falls back to pulling a base model from Hugging Face to ensure the UI remains active).*

## ⚠️ Responsible-Use Policy
This tool has a measured false-accusation rate. It is designed as an analytical aid and should **never** be used as sole evidence against a person for academic dishonesty or professional misconduct.

---
*Developed for MAIB AI 115 | NLP and Models | GANs in NLP*
