import torch
from document_classifier.train_nn import TextClassifier, encode
import argparse


def predict_text(text,model,vocab,target_names):
    ids = torch.tensor(encode(text,vocab),dtype=torch.long)
    ids = ids.unsqueeze(0)
    with torch.no_grad():
        scores = model(ids)
        predicted_label = scores.argmax(dim = 1).item()
    return target_names[predicted_label]
def load_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        )
    model = TextClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    vocab = checkpoint["vocab"]
    target_names = checkpoint["target_names"]
    return model,vocab, target_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help = "Document text")
    args = parser.parse_args()

    checkpoint_path  = "artifacts/text_classifier.pt"
   


    text = args.text
    model,vocab, target_names = load_model(checkpoint_path=checkpoint_path)

    print(text)
    print(predict_text(text,model,vocab,target_names))

if __name__ == "__main__":
    main()