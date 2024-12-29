from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Conversation history for maintaining context
conversation_history = []

model_name = "microsoft/Phi-3-mini-4k-instruct"
# Load the Hugging Face model (Ensure you have a model that fits your use case)
chatbot_model = pipeline("text-generation", model=model_name, trust_remote_code=True)

@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    global conversation_history
    conversation_history.clear()  # Reset conversation history
    return jsonify({"message": "Conversation history cleared."}), 200

@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get the user input from the request body
    user_input = request.json.get('message')
    
    if not user_input:
        return jsonify({'response': 'Please provide a message.'}), 400
    
    try:
        # Generate a response using the model
        response = generate_response(user_input)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f"Error: {str(e)}"}), 500

def generate_response(user_input):
    # Append user input to the conversation history
    conversation_history.append(
        {"role": "user", "content": user_input},
    )

    # Generate a response using the Hugging Face model
    try:
        result = chatbot_model(conversation_history, num_return_sequences=1, max_new_tokens=250)
        print(result)
        # Append assistant's response to the conversation history
        assistant_response = result[0]['generated_text'][-1]['content']
        conversation_history.append(
            {"role": "assistant", "content": assistant_response},
        )
        return assistant_response
    except Exception as e:
        raise Exception("Model generation failed") from e

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
