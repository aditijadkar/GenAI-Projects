
# Text Generation with GPT-2

Train a model to generate coherent and contextually relevant text based on a given prompt. Starting with GPT-2, a transformer model developed by OpenAI, fine-tune the model on a custom dataset to create text that mimics the style and structure of your training data.

This is a Flask-based Generative AI application that utilizes GPT-2 for text generation and includes a user-friendly interface.

## Features
- Text generation using GPT-2.
- User-friendly controls for adjusting parameters. Users can specify max_length, temperature, top_k, top_p, and the number of outputs
- Restart Button: Clear inputs and outputs for a fresh start without refreshing the page.
- Sentiment analysis of generated text.

## Project Structure
- `app.py`: Flask backend for text generation 
- `templates/index.html`: Frontend HTML file.
- `static/styles.css`: Styling for the user interface.

## How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install flask torch transformers
3. Run the Flask App:
   python app.py
4. Open your browser and visit http://127.0.0.1:5000
