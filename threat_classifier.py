"""
threat_classifier.py — Train severity classifier on RT_IOT2022.csv.
Uses BalancedRandomForest to handle 2500:1 class imbalance.

Install: pip install imbalanced-learn
"""

import numpy as np, pandas as pd, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    USE_BALANCED = True; print("Using BalancedRandomForestClassifier")
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    USE_BALANCED = False; print("imbalanced-learn not found, using standard RF")

FEATURES = ["flow_duration","fwd_pkts_tot","bwd_pkts_tot","fwd_pkts_per_sec",
    "bwd_pkts_per_sec","flow_pkts_per_sec","payload_bytes_per_second",
    "fwd_header_size_tot","bwd_header_size_tot","flow_FIN_flag_count",
    "flow_SYN_flag_count","flow_RST_flag_count","flow_PSH_flag_count",
    "flow_ACK_flag_count","fwd_pkts_payload.avg","bwd_pkts_payload.avg",
    "fwd_pkts_payload.std","bwd_pkts_payload.std","flow_pkts_payload.avg",
    "flow_pkts_payload.std","fwd_iat.avg","bwd_iat.avg","active.avg","idle.avg"]

HIGH  = ["ddos","dos","flood","syn_hping","exploit","ransomware","botnet","backdoor","injection","brute","password"]
MED   = ["scan","nmap","recon","probe","portscan","fingerprint","discovery","arp","spoof","mitm","os_","wipro","bulb"]
LOW   = ["mqtt","thing_speak","weather","normal","benign"]
NAMES = {0:"LOW/BENIGN",1:"MEDIUM",2:"HIGH"}

def _sev(at):
    v=str(at).strip().lower().replace(" ","_")
    if any(k in v for k in LOW): return 0
    if any(k in v for k in HIGH): return 2
    if any(k in v for k in MED): return 1
    return 1

df = pd.read_csv("RT_IOT2022.csv")
print(f"Loaded {len(df):,} rows")

sev_map = {v: _sev(v) for v in df["Attack_type"].unique()}
df["label"] = df["Attack_type"].map(sev_map).astype(int)

print(f"\nSeverity mapping:")
for v,s in sorted(sev_map.items(), key=lambda x:x[1]):
    print(f"  {str(v):30s} -> {NAMES[s]}  ({(df['Attack_type']==v).sum():,})")

avail = [f for f in FEATURES if f in df.columns]
print(f"\nUsing {len(avail)} features")

X = df[avail].replace([np.inf,-np.inf],np.nan).fillna(0)
y = df["label"].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler(); X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)

if USE_BALANCED:
    clf = BalancedRandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=3,
                                         sampling_strategy="auto", replacement=False, n_jobs=-1, random_state=42)
else:
    clf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=3,
                                 n_jobs=-1, random_state=42, class_weight="balanced")

print("\nTraining..."); clf.fit(X_tr_s, y_tr)
report = classification_report(y_te, clf.predict(X_te_s), target_names=["Low/Benign","Medium","High"])
print(f"\n{report}")

print("Per-class accuracy:")
for i,name in enumerate(["Low/Benign","Medium","High"]):
    mask = y_te==i
    if mask.sum()>0: print(f"  {name:15s}: {(clf.predict(X_te_s)[mask]==i).mean():.3f}  ({mask.sum():,} samples)")

imp = pd.Series(clf.feature_importances_, index=avail).sort_values(ascending=False)
print("\nTop 10 features:")
for f,s in imp.head(10).items(): print(f"  {f:35s} {s:.4f}")

with open("threat_report.txt","w") as f:
    f.write(f"Classifier: {'BalancedRF' if USE_BALANCED else 'RF'}\n\n{report}\n")
    for feat,sc in imp.items(): f.write(f"  {feat:35s} {sc:.4f}\n")

bundle = {"classifier":clf, "scaler":scaler, "feature_names":avail,
          "rl_norm_params":{"latency_min":float(np.log1p(df["payload_bytes_per_second"]).min()),
                            "latency_max":float(np.log1p(df["payload_bytes_per_second"]).max()),
                            "pktload_min":float((df["fwd_pkts_tot"]+df["bwd_pkts_tot"]).min()),
                            "pktload_max":float((df["fwd_pkts_tot"]+df["bwd_pkts_tot"]).max())},
          "n_classes":3, "class_labels":list(clf.classes_), "severity_map":sev_map}

with open("threat_model.pkl","wb") as f: pickle.dump(bundle,f)
print(f"\nSaved threat_model.pkl + threat_report.txt")
