# Import necessary libraries
from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
from comet_ml import Experiment
from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()

comet_api_key = os.getenv("COMET_API_KEY")


# Load Rotten Tomatoes dataset
raw_datasets = load_dataset("rotten_tomatoes")
train_dataset = raw_datasets["train"]

# Initialize Flask app
app = Flask(__name__)

# Load BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Initialize Comet experiment
experiment = Experiment(api_key=comet_api_key, project_name="text-classifier-app")

# Define a function for text classification
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = logits.softmax(dim=1)
    return probabilities.detach().numpy().tolist()[0]

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for text classification
@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        text = request.form['text']
        # Track experiment using Comet
        with experiment.train():
            probabilities = classify_text(text)
            # Log input and output to Comet
            experiment.log_text("input_text", text)
            experiment.log_metrics({"probabilities": probabilities})
        return render_template('result.html', text=text, probabilities=probabilities)

# Define route for dataset examples
@app.route('/dataset_examples')
def dataset_examples():
    examples = train_dataset.shuffle(seed=42).select([0, 1])["text"][:5].tolist()
    return render_template('dataset_examples.html', examples=examples)

if __name__ == '__main__':
    app.run(debug=True)
