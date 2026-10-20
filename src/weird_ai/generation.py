import torch


def text_to_token_ids(text, tokenizer):
    """
    Convert text into a tensor of token IDs.
    """
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs back into text.
    """
    if token_ids.dim() > 1:
        token_ids = token_ids.squeeze(0)
    return tokenizer.decode(token_ids.tolist())


def generate_text_simple(model, input_ids, max_new_tokens, context_size):
    """
    Generate text one token at a time using greedy decoding.
    """
    for _ in range(max_new_tokens):
        input_ids = input_ids[:, -context_size:]

        with torch.no_grad():
            logits = model(input_ids)

        logits = logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        input_ids = torch.cat((input_ids, next_token_id), dim=-1)

    return input_ids


def generate_and_print_sample(
    model,
    tokenizer,
    device,
    start_context,
    context_size,
    max_new_tokens=50,
):
    """
    Generate and print a sample text output.
    """
    model.eval()

    token_ids = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            input_ids=token_ids,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
        )

    generated_text = token_ids_to_text(token_ids, tokenizer)
    print(generated_text.replace("\n", " "))