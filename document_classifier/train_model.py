



from document_classifier.data import load_data_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
def main():

    train_texts, val_texts, train_labels, val_labels, target_names = load_data_split()

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)


    model = LogisticRegression(max_iter= 1000)
    model.fit(X_train,train_labels)
  
    print(" train acc: ", model.score(X_train,train_labels))
    print(" val acc: ",   model.score(X_val,val_labels))


    pred_labels = model.predict(X_val)
    print(classification_report(val_labels,pred_labels,target_names=target_names))


    c_mat = confusion_matrix(val_labels, pred_labels)
    print(c_mat)
if __name__ == "__main__":
    main()