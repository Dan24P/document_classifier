
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
from sklearn.model_selection import train_test_split



def load_data_split():
    dataset = fetch_20newsgroups(subset= "train",categories=["sci.space","rec.sport.baseball","comp.graphics"],remove= ("headers","footers","quotes"))
    texts = []
    labels = []
    for text,label in zip(dataset.data,dataset.target):
        if text.strip():
            texts.append(text)
            labels.append(label)
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts,labels  ,train_size= 0.8,random_state= 7,stratify=labels)



    return train_texts, val_texts, train_labels, val_labels, dataset.target_names

