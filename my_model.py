from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chat(user_input):
	input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
	chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
	return tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

user_input = "Does money buy happiness?"
response = chat(user_input)
print(response)

tokens = tokenizer.tokenize("Who will win the Elections this year?")
print(tokens)