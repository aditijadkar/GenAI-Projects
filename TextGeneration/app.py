from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch

# Initialize Flask App
app = Flask(__name__)

# Load GPT-2 Model and Tokenizer
print("Loading GPT-2 model and tokenizer...")
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate Text Function
def generate_text(prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95, num_outputs=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=num_outputs,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    results = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return results

# Sentiment Analysis (Optional)
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    max_length = int(data.get("max_length", 100))
    temperature = float(data.get("temperature", 1.0))
    top_k = int(data.get("top_k", 50))
    top_p = float(data.get("top_p", 0.95))
    num_outputs = int(data.get("num_outputs", 1))
    
    responses = generate_text(prompt, max_length, temperature, top_k, top_p, num_outputs)
    return jsonify({"responses": responses})

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")
    analysis = analyze_sentiment(text)
    return jsonify({"sentiment": analysis})

if __name__ == "__main__":
    app.run(debug=True)
