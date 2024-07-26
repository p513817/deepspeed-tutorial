# tokenize dataset
def preprocess_with_tokenizer(tokenizer, examples):
    def extract_text(messages):
        return " ".join([msg["content"] for msg in messages])

    inputs = [
        "summarize: " + prompt + " " + extract_text(messages)
        for prompt, messages in zip(examples["prompt"], examples["messages"])
    ]
    model_inputs = tokenizer(
        inputs, max_length=256, padding="max_length", truncation=True
    )
    labels = tokenizer(
        text_target=examples["category"],
        max_length=256,
        padding="max_length",
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
