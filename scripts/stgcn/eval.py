import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from graph_hand import Graph
from dataset import KeypointDataset
from model import STGCN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", type=str, default=r"dataset/processed/labels.csv")
    ap.add_argument("--ckpt", type=str, default=r"checkpoints_full/best_stgcn_full.pt")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ckpt = torch.load(args.ckpt, map_location=device)
    use_xyz = ckpt["use_xyz"]
    in_channels = ckpt["in_channels"]
    num_class = ckpt["num_class"]

    graph = Graph(strategy="spatial", max_hop=1)
    A = graph.A

    model = STGCN(num_class=num_class, in_channels=in_channels, A=A, num_joints=21).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_ds = KeypointDataset(args.labels_csv, split="test", use_xyz=use_xyz)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(y.tolist())

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
