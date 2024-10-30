from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer


app = Flask(__name__)

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


@app.route('/chat', methods=['POST'])
def chat():
	data = request.get_json()
	user_input = data['input']
	input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
	chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
	response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
	return jsonify({'response': response})


@app.route('/', methods=['GET'])
def home():
	return "<b>Hello Algorizin</b>"

if __name__ == '__main__':
	app.run(debug=True)
