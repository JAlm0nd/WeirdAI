import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = "Akwirw ier"

token_ids = tokenizer.encode(text)
print(token_ids)

for token_id in token_ids:
    print(token_id, repr(tokenizer.decode([token_id])))

print(tokenizer.decode(token_ids))