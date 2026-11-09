from sklearn.datasets import fetch_20newsgroups
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
def main():
    print("Ready to inspect data" )
    dataset = fetch_20newsgroups(subset= "train",categories=["sci.space","rec.sport.baseball","comp.graphics"],remove= ("headers","footers","quotes"))

    texts = []
    labels = []
    for text,label in zip(dataset.data,dataset.target):
        if text.strip():
            texts.append(text)
            labels.append(label)


    counts = Counter(labels)
    for label, count in counts.items():
        print(dataset.target_names[label],count)




    print(len(texts),len(labels))



    train_texts, val_texts, train_labels, val_labels = train_test_split(texts,labels  ,train_size= 0.8,random_state= 7,stratify=labels)
    print(len(train_texts),len(train_labels))
    print(len(val_texts),len(val_labels))

    count_train = Counter(train_labels)
    count_val = Counter(val_labels)
    print("train counts")
    for label, count in count_train.items():
            print(dataset.target_names[label],count)

    print("val counts")
    for label, count in count_val.items():
            print(dataset.target_names[label],count)


    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    
    print(X_train.shape)
    print(X_val.shape)
    print(vectorizer.get_feature_names_out()[:10])

    
if __name__ == "__main__":
    main()