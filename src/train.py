# src/train.py
import torch, torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def train(model, loaders, epochs=15, lr=3e-4, wd=1e-4, device="cpu"):
    train_ld, val_ld, _ = loaders
    model.to(device)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min',
                                                       factor=0.5, patience=3)
    crit  = nn.CrossEntropyLoss()
    best_acc = 0
    for ep in range(1, epochs+1):
        # --- train
        model.train(); tloss=0; tcorrect=0; tsamp=0
        for x,y in tqdm(train_ld, desc=f"Train {ep}"):
            x,y = x.to(device), y.to(device)
            optim.zero_grad()
            out = model(x); loss = crit(out,y); loss.backward(); optim.step()
            tloss += loss.item()*x.size(0)
            tcorrect += (out.argmax(1)==y).sum().item(); tsamp += x.size(0)
        # --- val
        vloss, vacc = evaluate(model, val_ld, crit, device)
        sched.step(vloss)
        print(f"[E{ep:02d}] train {tcorrect/tsamp:.3%} | val {vacc:.3%} | lr {optim.param_groups[0]['lr']:.1e}")
        if vacc>best_acc:
            best_acc=vacc; torch.save(model.state_dict(), "best.pt")
    return best_acc

@torch.inference_mode()
def evaluate(model, loader, crit, device):
    model.eval(); loss=0; correct=0; total=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        out=model(x); loss+=crit(out,y).item()*x.size(0)
        correct+=(out.argmax(1)==y).sum().item(); total+=x.size(0)
    return loss/total, correct/total
