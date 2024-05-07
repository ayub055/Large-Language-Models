import re

def vocabulary(data = 'the-verdict.txt'):
    with open(data, "r", encoding = "utf-8") as f:
        raw_text = f.read()
    
    preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed_text = [item for item in preprocessed_text if item.strip()]
    unique_words = sorted(set(preprocessed_text))    
    unique_words.extend(["<EOS>", "<UNK>"])
    vocab = {word:id for id, word in enumerate(unique_words)}
    # print(len(unique_words), len(preprocessed_text), len(vocab))
    return vocab
    
class SimpleTokenizer:
    def __init__(self, vocab):
        self.token_to_ids = vocab
        self.ids_to_token = {idx : token for token, idx in vocab.items()}
    
    def encode(self, text):
        preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed_text = [token for token in preprocessed_text if token.strip()]
        preprocessed_text = [token if token in self.token_to_ids else "<UNK>" for token in preprocessed_text]
        token_ids = [self.token_to_ids[token] for token in preprocessed_text]
        return token_ids
        
    def decode(self, token_ids) :
        text = " ".join([self.ids_to_token[idx] for idx in token_ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
if __name__ == "__main__": 
    
    vocab = vocabulary()
    tokenizer = SimpleTokenizer(vocab=vocab)
    
    text = "Hello, world. This, is a test."
    enc = tokenizer.encode(text)
    dec = tokenizer.decode(enc)
    print(enc)
    print(dec)
    
    
    
    
    