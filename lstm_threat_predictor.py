"""
lstm_threat_predictor.py — LSTM for future threat prediction.

Train:   python lstm_threat_predictor.py
Use:     python Train_dqn.py --use-lstm
"""

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEQ_LEN=20; HORIZON=5; HIDDEN_DIM=64; NUM_LAYERS=2; DROPOUT=0.2
BATCH_SIZE=256; EPOCHS=50; LR=1e-3; SEED=42
torch.manual_seed(SEED); np.random.seed(SEED)

def load_threat_sequence(csv="state_vectors.csv"):
    return pd.read_csv(csv)["threat_score"].values.astype(np.float32)

def make_sequences(ts, seq_len, horizon):
    X, y = [], []
    for i in range(len(ts)-seq_len-horizon+1):
        X.append(ts[i:i+seq_len]); y.append(ts[i+seq_len+horizon-1])
    return np.array(X,dtype=np.float32)[:,:,np.newaxis], np.array(y,dtype=np.float32)

class ThreatLSTM(nn.Module):
    def __init__(self, hid=64, layers=2, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hid, layers, batch_first=True, dropout=drop if layers>1 else 0)
        self.head = nn.Sequential(nn.Linear(hid,32), nn.ReLU(), nn.Linear(32,1), nn.Sigmoid())
    def forward(self, x):
        out, _ = self.lstm(x); return self.head(out[:,-1,:]).squeeze(-1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = load_threat_sequence(); X, y = make_sequences(ts, SEQ_LEN, HORIZON)
    split = int(len(X)*0.85)
    tr_dl = DataLoader(TensorDataset(torch.from_numpy(X[:split]),torch.from_numpy(y[:split])), BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(X[split:]),torch.from_numpy(y[split:])), BATCH_SIZE)
    model = ThreatLSTM(HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR); crit = nn.MSELoss()
    print(f"Training LSTM ({len(X):,} seqs, device={device})")
    best = float("inf")
    for ep in range(1, EPOCHS+1):
        model.train(); tl=0
        for xb,yb in tr_dl:
            xb,yb=xb.to(device),yb.to(device); loss=crit(model(xb),yb)
            opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item()*len(xb)
        model.eval(); vl=0
        with torch.no_grad():
            for xb,yb in va_dl: xb,yb=xb.to(device),yb.to(device); vl+=crit(model(xb),yb).item()*len(xb)
        tl/=len(tr_dl.dataset); vl/=len(va_dl.dataset)
        if vl<best: best=vl; torch.save(model.state_dict(),"lstm_threat.pt")
        if ep%5==0 or ep==1: print(f"  Ep {ep:>3}  train={tl:.6f}  val={vl:.6f}  {'*' if vl==best else ''}")
    print(f"Best val MSE: {best:.6f}. Saved lstm_threat.pt")

class ThreatPredictor:
    def __init__(self, path="lstm_threat.pt", hid=HIDDEN_DIM, layers=NUM_LAYERS):
        self.model = ThreatLSTM(hid, layers, 0.0)
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model.eval(); self.seq_len = SEQ_LEN
    def predict(self, recent):
        arr = np.array(list(recent), dtype=np.float32)
        if len(arr)<self.seq_len: arr=np.pad(arr,(self.seq_len-len(arr),0),mode="mean")
        arr = arr[-self.seq_len:][np.newaxis,:,np.newaxis]
        with torch.no_grad(): return float(self.model(torch.from_numpy(arr)).item())
