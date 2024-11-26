from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Step 1: Specify the model name
model_name = "facebook/blenderbot-400M-distill"

# Step 2: Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 3: Initialize the conversation history
conversation_history = []

# Step 4: Interaction loop for the chatbot
while True:
    # Step 5.2: Encode conversation history as a string
    history_string = "\n".join(conversation_history)
    
    # Step 5.3: Fetch prompt from the user
    input_text = input("> ")
    
    # Exit the chat if the user types "exit"
    if input_text.lower() == "exit":
        print("Goodbye!")
        break

    # Step 5.4: Tokenization of user prompt and chat history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
    
    # Step 5.5: Generate the response from the model
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,         # Limit response length
        do_sample=True,            # Enable sampling-based generation
        temperature=0.7,           # Control randomness
        top_k=30,                  # Use tighter top-k sampling
        top_p=0.8,                 # Use tighter nucleus sampling
        repetition_penalty=2.5     # Penalize repetition further
    )
    
    # Step 5.6: Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(response)
    
    # Step 5.7: Update the conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
    conversation_history = conversation_history[-6:]  # Limit history to the last 6 exchanges

