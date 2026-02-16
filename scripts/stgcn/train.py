import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from graph_hand import Graph
from dataset import KeypointDataset
from model import STGCN

def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", type=str, default=r"dataset/processed/labels.csv")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--use_xyz", action="store_true", help="use x,y,z (default x,y)")
    ap.add_argument("--save_dir", type=str, default="checkpoints_full")
    ap.add_argument("--num_class", type=int, default=35)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    graph = Graph(strategy="spatial", max_hop=1)
    A = graph.A  # (K,V,V)

    in_channels = 3 if args.use_xyz else 2
    model = STGCN(num_class=args.num_class, in_channels=in_channels, A=A, num_joints=21).to(device)

    train_ds = KeypointDataset(args.labels_csv, split="train", use_xyz=args.use_xyz)
    val_ds   = KeypointDataset(args.labels_csv, split="val", use_xyz=args.use_xyz)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        for x, y in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += accuracy(logits, y)
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        train_acc = running_acc / max(1, n_batches)

        # Validation
        model.eval()
        v_acc = 0.0
        v_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                v_acc += accuracy(logits, y)
                v_batches += 1

        val_acc = v_acc / max(1, v_batches)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            ckpt_path = os.path.join(args.save_dir, "best_stgcn_full.pt")
            torch.save({
                "model": model.state_dict(),
                "use_xyz": args.use_xyz,
                "num_class": args.num_class,
                "in_channels": in_channels
            }, ckpt_path)
            print("✅ Saved best checkpoint:", ckpt_path)

    print("Best val_acc:", best_val)

if __name__ == "__main__":
    main()
