# 🎭 Multimodal Emotion Detection using Late Fusion

A multimodal AI system that combines **text-based** and **facial expression** emotion detection using a Late Fusion architecture.

---

## 🧠 How It Works

This system uses two models:

- **BERT** (RoBERTa GoEmotions) — analyzes text and detects **27 emotions**
- **DeepFace** — analyzes facial images and detects **7 emotions**

Both models run independently and their outputs are combined using **Late Fusion (Weighted Average)** to produce a final emotion prediction.

**Pipeline:**
```
Text → BERT → 27 Emotion Probabilities ──┐
                                          ├── Weighted Avg → Final Emotion
Image → DeepFace → 7 Emotion Probabilities ┘
```

---

## ⚙️ Tech Stack

- Python
- HuggingFace Transformers (RoBERTa GoEmotions)
- DeepFace
- Google Colab

---

## 🚀 How to Run

1. Open the notebook in Google Colab
2. Install dependencies:
```bash
pip install deepface transformers
```
3. Run all cells
4. Provide a text input and a face image
5. Get the fused emotion output

---

## 📊 Results

The model outputs a weighted average of both models:
- BERT weight: **60%**
- DeepFace weight: **40%**

---

## 🔮 Future Scope

- **Meta-Classifier** — replace fixed weights with a trained neural network that learns optimal fusion automatically
- **Web UI** — interactive interface for real-time emotion detection
- **Video Support** — frame-by-frame facial emotion analysis
- **Audio Branch** — add speech/tone analysis as a third modality

---

## 📁 Dataset

- Text emotions: [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) by Google
- Facial emotions: DeepFace pre-trained model
