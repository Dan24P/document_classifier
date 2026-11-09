from document_classifier.data import load_data_split
import torch
from torch import nn
import re, random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from pathlib import Path

def build_vocab(texts):
    vocab = {
        "<pad>":0,
        "<unk>":1
    }
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab
def tokenize(text):
    return re.findall(r"\b\w+\b",text.lower())

def encode(text,vocab):
    ids = []
    tokens = tokenize(text)
    for token in tokens:
        ids.append( vocab.get(token,vocab["<unk>"]) )
    if not ids:
        ids.append(vocab["<unk>"])
    return ids


class TextClassifier(nn.Module):
    def __init__(self, vocab_size,embedding_dim,num_classes,pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim= embedding_dim,
            padding_idx=pad_idx
        )
        self.classifier = nn.Linear(
            in_features= embedding_dim,
            out_features= num_classes
        )
        self.pad_idx = pad_idx
    def forward(self,ids):
        vectors = self.embedding(ids)
        mask = ids != self.pad_idx
        masked_vectors = vectors * mask.unsqueeze(-1)
        sums = masked_vectors.sum(dim = 1)
        lengths = mask.sum(dim = 1 ,keepdim = True)
        document_vector = sums / lengths
        scores =self.classifier(document_vector)
        return scores
def train_epoch(model,loader,optimizer,loss_function):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for ids, target in loader:
        optimizer.zero_grad()
        scores = model(ids)
        loss = loss_function(scores,target)
        loss.backward()
        
        optimizer.step()
        batch_size = len(target)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss/total_examples


def evaluate(model,samples,loss_function):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for ids, target in samples:
            
            scores = model(ids.unsqueeze(0))[0]
            loss = loss_function(scores,target)
            total_loss +=  loss.item()

            prediction = scores.argmax().item()
            if(prediction == target.item()):
                correct += 1
    return total_loss / len(samples) , correct /len(samples)

def collate_batch(samples):
    sequences = []
    targets = []
    for id, target in samples:
        sequences.append(id)
        targets.append(target)
    padded_ids = pad_sequence(
    sequences,
    batch_first=True,
    padding_value=0,
    )
    targets_tensor = torch.stack(targets)
    return padded_ids, targets_tensor
def main():
    torch.manual_seed(7)
    random.seed(7)
    Path("artifacts").mkdir(exist_ok=True)
    train_texts, val_texts, train_labels, val_labels, target_names = load_data_split()
    vocab = build_vocab(train_texts)
    model_config = {
    "vocab_size": len(vocab),
    "embedding_dim": 8,
    "num_classes": len(target_names),
    "pad_idx": vocab["<pad>"],
    }
    model = TextClassifier(
            **model_config
        )
    train_samples = []
    val_samples = []
    for text, label in zip(train_texts,train_labels):
        ids = torch.tensor(encode(text,vocab),dtype=torch.long)
        target =  torch.tensor(label,dtype=torch.long)
        train_samples.append((ids,target))
    train_loader = DataLoader(
        train_samples,
        batch_size = 32,
        shuffle = True,
        collate_fn = collate_batch,
    )
    #batch_ids, batch_targets = next(iter(train_loader))
    for text, label in zip(val_texts,val_labels):
        ids = torch.tensor(encode(text,vocab),dtype=torch.long)
        target =  torch.tensor(label,dtype=torch.long)
        val_samples.append((ids,target))


    loss_function = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
    best_val_loss = float("inf")
    for i in range(80):
        #random.shuffle(train_samples)
        train_loss = train_epoch(model,train_loader,optimizer,loss_function)
        val_loss, val_acc = evaluate(model,val_samples,loss_function)
        print(i,"train loss", train_loss, "val loss", val_loss, "val acc", val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint ={
                "model_state_dict": model.state_dict(),
                "vocab": vocab ,
                  "target_names": target_names ,
                "model_config": model_config,
                "epoch": i,
                "val_loss": val_loss,
                "val_acc": val_acc
            }
            torch.save(checkpoint, "artifacts/text_classifier.pt")

if __name__ == "__main__":
    main()