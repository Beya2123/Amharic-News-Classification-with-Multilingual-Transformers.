# Amharic-News-Classification-with-Multilingual-Transformers.
Amharic News Classification with Multilingual Transformers. This repository contains three transformer-based models for classifying Amharic news articles. The models leverage state-of-the-art multilingual language models to effectively handle the Amharic language.
Amharic News Classification with Multilingual Transformers
This repository contains three transformer-based models for classifying Amharic news articles. The models leverage state-of-the-art multilingual language models to effectively handle the Amharic language.

📋 Project Overview
This project implements and compares three different multilingual transformer models for Amharic news classification:

mBERT (Multilingual BERT)

XLM-RoBERTa (Cross-lingual Language Model)

Afro-XLM-RoBERTa (African-focused multilingual model)

These models are trained to classify Amharic news articles into various categories (politics, sports, technology, etc.) using the Amharic News Dataset.

🚀 Models
1. mBERT (Multilingual BERT)
Model: bert-base-multilingual-cased

Description: Google's BERT model trained on 104 languages including Amharic

File: mBert.py

Key Features:

12-layer, 768-hidden, 12-heads, 110M parameters

Trained on Wikipedia data

Supports tokenization for Amharic

2. XLM-RoBERTa
Model: xlm-roberta-base

Description: Facebook's cross-lingual model trained on 100 languages

File: XLM-roberta.py

Key Features:

12-layer, 768-hidden, 12-heads, 270M parameters

Trained on CommonCrawl data

Specifically designed for cross-lingual tasks

3. Afro-XLM-Roberta
Model: castorini/afroxlmr-base

Description: XLM-RoBERTa model fine-tuned on African languages

File: Afro-xlmr-base.py

Key Features:

Optimized for African languages including Amharic

Better performance on low-resource languages

Includes language-specific adaptations

📊 Dataset
The models are trained on the Amharic News Dataset which contains:

Amharic news articles with category labels

Multiple categories (e.g., politics, sports, business, technology)

Preprocessed text data with Amharic character preservation

🛠️ Installation
Prerequisites
Python 3.7+

PyTorch 1.8+

Transformers library

Install Dependencies
bash
pip install -r requirements.txt
Or manually install:

bash
pip install torch transformers pandas scikit-learn numpy matplotlib seaborn
📁 Project Structure
text
amharic-news-classification/
├── mBert.py                    # mBERT model implementation
├── XLM-roberta.py              # XLM-RoBERTa model implementation
├── Afro-xlmr-base.py           # Afro-XLM-Roberta model implementation
├── utils.py                    # Utility functions
├── config.py                   # Configuration settings
├── requirements.txt            # Dependencies
├── data/
│   ├── amharic_news_dataset.csv  # Dataset file
│   └── processed/               # Processed data
├── models/                     # Saved model weights
│   ├── mbert_model/
│   ├── xlmr_model/
│   └── afroxlmr_model/
├── results/                    # Evaluation results
└── README.md                   # This file
🚀 Quick Start
1. Prepare Dataset
Place your Amharic news dataset CSV file in the data/ directory. The file should contain at least two columns: text content and category labels.

2. Run Individual Models
mBERT Model:

bash
python mBert.py --data_path data/amharic_news_dataset.csv --epochs 5 --batch_size 16
XLM-RoBERTa Model:

bash
python XLM-roberta.py --data_path data/amharic_news_dataset.csv --epochs 5 --batch_size 16
Afro-XLM-R Model:

bash
python Afro-xlmr-base.py --data_path data/amharic_news_dataset.csv --epochs 5 --batch_size 16
3. Training Parameters
Common arguments for all models:

bash
--data_path        # Path to dataset CSV file
--epochs           # Number of training epochs (default: 5)
--batch_size       # Batch size (default: 16)
--learning_rate    # Learning rate (default: 2e-5)
--max_length       # Maximum sequence length (default: 256)
--test_size        # Test set proportion (default: 0.2)
--model_save_path  # Path to save trained model
📈 Results Comparison
The models can be compared based on:

Accuracy: Overall classification accuracy

F1-Score: Weighted F1 score for imbalanced classes

Training Time: Time taken for training

Inference Speed: Time taken for predictions

Resource Usage: GPU/CPU memory consumption

Expected Performance
mBERT: Fast training, good baseline performance

XLM-RoBERTa: Better accuracy, larger model size

Afro-XLM-R: Best performance for Amharic, specialized for African languages

🔧 Customization
Modify Model Configuration
Edit config.py to change default parameters:

python
# Training parameters
TRAINING_CONFIG = {
    'batch_size': 16,
    'epochs': 5,
    'learning_rate': 2e-5,
    'max_length': 256,
    'test_size': 0.2
}

# Model paths
MODEL_PATHS = {
    'mbert': 'bert-base-multilingual-cased',
    'xlmr': 'xlm-roberta-base',
    'afroxlmr': 'castorini/afroxlmr-base'
}
Add Custom Preprocessing
Edit utils.py to add custom text preprocessing:

python
def preprocess_amharic_text(text):
    """
    Custom preprocessing for Amharic text
    """
    # Remove English characters if needed
    # Normalize Amharic characters
    # Remove specific patterns
    return processed_text
📊 Evaluation
Each model script includes comprehensive evaluation:

Accuracy score

F1-score (weighted and per-class)

Classification report

Confusion matrix (optional)

Model predictions on sample texts

🎯 Inference
Load and Use Trained Model
python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load mBERT model
model = AutoModelForSequenceClassification.from_pretrained('models/mbert_model')
tokenizer = AutoTokenizer.from_pretrained('models/mbert_model')

# Make prediction
def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)
    return prediction
📈 Performance Tips
For Limited Resources: Use mBERT (smallest model)

For Best Accuracy: Use Afro-XLM-Roberta

For Balanced Performance: Use XLM-RoBERTa

For Faster Training: Reduce max_length and batch_size

For Better Results: Increase epochs and use data augmentation

🐛 Troubleshooting
Common Issues:
CUDA Out of Memory:

bash
# Reduce batch size
python model.py --batch_size 8

# Reduce sequence length
python model.py --max_length 128
Dataset Format Issues:

Ensure CSV has correct encoding (UTF-8)

Check column names in the dataset

Remove null values from the dataset

Model Download Issues:

python
# Use local model path
model = AutoModel.from_pretrained('./local_model')
📚 References
mBERT Paper

XLM-RoBERTa Paper

Afro-XLM-Roberta Repository

Hugging Face Transformers

Amharic Language Processing

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

🙏 Acknowledgments
The Hugging Face team for the transformers library

Contributors to the Amharic NLP resources

Researchers working on African language processing

📧 Contact
For questions or feedback, please open an issue in the GitHub repository.

Note: Ensure you have sufficient computational resources (GPU recommended) for training these models. The Afro-XLM-Roberta model may require additional disk space for downloading the pretrained weights.

