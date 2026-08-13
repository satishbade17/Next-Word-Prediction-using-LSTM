Next Word Prediction using LSTM

A Deep Learning–based Next Word Prediction project using LSTM (Long Short-Term Memory) and the WikiText-2 dataset.

🚀 Project Overview

This project predicts the most likely next word based on the input text. The model is trained on real-world text data and provides Top-5 next-word suggestions through an interactive Streamlit web application.

🛠️ Technologies Used
🐍 Python
🧠 LSTM — Long Short-Term Memory
🤖 TensorFlow / Keras
🔤 NLP — Natural Language Processing
📚 WikiText-2 Dataset
📊 NumPy
🌐 Streamlit
💾 Pickle
✨ Features
Next-word prediction using LSTM
Text preprocessing and tokenization
Sequence generation and padding
Top-5 word suggestions
Model evaluation
Saved trained model and tokenizer
Interactive Streamlit interface
📂 Dataset

The model uses the WikiText-2 dataset, with:

train.txt
test.txt

## Dataset

The WikiText-2 dataset files are not included in this repository
because the dataset files are larger than GitHub's recommended
file size limit.

Download the dataset separately and place:

- train.txt
- test.txt

in the project folder before running the training script.
▶️ How to Run

Install the required libraries:

pip install tensorflow streamlit numpy

Train the model:

python train_model.py

Run the Streamlit application:

streamlit run app.py
📌 Example
Input:
the first time

Next Word Suggestions:
1. ...
2. ...
3. ...
4. ...
5. ...
🎯 Objective

The main objective of this project is to understand how LSTM networks can process sequential text data and predict the next word, providing a practical introduction to language modeling and NLP.
#Python #LSTM #NLP #DeepLearning #MachineLearning #TensorFlow #Keras #Streamlit #ArtificialIntelligence #NextWordPrediction
