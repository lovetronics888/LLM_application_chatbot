from flask import Flask, render_template
from flask_cors import CORS		# newly added
from flask import request
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Setup Model and Tokenizer
# This model is great for basic conversation and isn't too heavy for most computers
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

app = Flask(__name__)
CORS(app)				# newly added

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
   
    # 2. Initialize Memory
    # This list will store your conversation so the bot remembers context
    conservation_history = []

    print("--- Chatbot Activated ---")
    print("(Type 'quit' or 'exit' to end the chat)")

    while True:
        #read prompt from HTTP request body
        data = request.get_data(as_text=True)
        data = json.loads(data)
        input_text = data['prompt']
        # 3. Get User Input
        #user_input = input("You: ")

        # 4. The 'Safety Switch' - check if user wants to leave
        if input_text.lower() in ["quit", "exit"]:
            print("Bot: Goodbye! Have a great day.")
            break

        # 5. Build the History String
        # Combines all old messages into one block of text
        history_string = "\n".join(conservation_history)

        # 6. Create Full Context
        # If history exists, add a newline before the new message; otherwise, just use the message
        if conservation_history:
            full_input = history_string + "\n" + input_text 
        else:
             full_input = input_text

        # 7. Tokenize Input
        # Converts your text into numbers (tensors) the AI can process
        # Add truncation=True and max_length=128
        inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=128)


        # 8. Generate Response
        # no_repeat_ngram_size=3 prevents the bot from repeating "doing doing doing"
        outputs = model.generate(**inputs, 
        max_new_tokens=50, 
        no_repeat_ngram_size=3, 
        do_sample=True, 
        top_k=50, 
        top_p=0.95)
                
    
        # 9. Decode and Clean Output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print(f"Bot: {response}")

        # 10. Update Memory
        # Save both parts of the conversation for the next turn
        conservation_history.append(input_text)
        conservation_history.append(response)

        # 11. Keep history manageable (last 10 messages)
        if len(conservation_history) > 10:
            conservation_history = conservation_history[-10:]
        return response

if __name__ == '__main__':
    app.run()
