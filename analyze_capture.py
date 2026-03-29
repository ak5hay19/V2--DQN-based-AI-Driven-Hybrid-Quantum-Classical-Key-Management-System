"""
analyze_capture.py — Analyze .pcap / .csv files with trained RL agent.

Usage:
    python analyze_capture.py traffic.pcap --report
    python analyze_capture.py RT_IOT2022.csv --max-flows 500 --balanced --report
    python analyze_capture.py demo_traffic.pcap --report --dqn
"""

import argparse, os, sys, numpy as np, pandas as pd, pickle
from collections import deque
from hybrid_env import ACTION_NAMES, NUM_ACTIONS

parser = argparse.ArgumentParser(description="SmartKeyNet Capture Analysis")
parser.add_argument("input_file")
parser.add_argument("--report", action="store_true")
parser.add_argument("--max-flows", type=int, default=500)
parser.add_argument("--balanced", action="store_true")
parser.add_argument("--use-lstm", action="store_true")
parser.add_argument("--dqn", action="store_true")
parser.add_argument("--qtable", action="store_true")
args = parser.parse_args()


def extract_pcap(path, max_flows):
    try: from scapy.all import PcapReader, IP, TCP, UDP
    except: print("ERROR: pip install scapy"); sys.exit(1)
    from collections import defaultdict
    flows = defaultdict(lambda: {"fwd":0,"bwd":0,"fb":0,"bb":0,"syn":0,"fin":0,"rst":0,"psh":0,"ack":0,
                                  "fp":[],"bp":[],"fi":[],"bi":[],"t0":None,"t1":None,"src":None,"dst":None})
    keys = set(); count = 0
    for pkt in PcapReader(path):
        if IP not in pkt: continue
        count += 1; ip = pkt[IP]; sp=dp=0
        if TCP in pkt: sp,dp=pkt[TCP].sport,pkt[TCP].dport
        elif UDP in pkt: sp,dp=pkt[UDP].sport,pkt[UDP].dport
        if (ip.src,sp)<=(ip.dst,dp): k=(ip.src,ip.dst,sp,dp,ip.proto); fwd=True
        else: k=(ip.dst,ip.src,dp,sp,ip.proto); fwd=False
        keys.add(k)
        if len(keys)>max_flows: break
        f=flows[k]; ts=float(pkt.time); pl=max(0,len(pkt)-40)
        if f["t0"] is None: f["t0"]=ts; f["src"]=ip.src if fwd else ip.dst; f["dst"]=ip.dst if fwd else ip.src
        f["t1"]=ts
        if fwd: f["fwd"]+=1; f["fb"]+=len(pkt); f["fp"].append(pl)
        else: f["bwd"]+=1; f["bb"]+=len(pkt); f["bp"].append(pl)
        if TCP in pkt:
            fl=pkt[TCP].flags
            if fl&0x02: f["syn"]+=1
            if fl&0x01: f["fin"]+=1
            if fl&0x04: f["rst"]+=1
            if fl&0x08: f["psh"]+=1
            if fl&0x10: f["ack"]+=1
    print(f"  {count:,} packets -> {len(flows):,} flows")
    records = []
    for k,f in flows.items():
        dur=max(f["t1"]-f["t0"],0.001); tot=f["fwd"]+f["bwd"]; tb=f["fb"]+f["bb"]
        fp=f["fp"] or [0]; bp=f["bp"] or [0]
        records.append({"flow_duration":dur,"fwd_pkts_tot":f["fwd"],"bwd_pkts_tot":f["bwd"],
            "fwd_pkts_per_sec":f["fwd"]/dur,"bwd_pkts_per_sec":f["bwd"]/dur,"flow_pkts_per_sec":tot/dur,
            "payload_bytes_per_second":tb/dur,"fwd_header_size_tot":f["fwd"]*40,"bwd_header_size_tot":f["bwd"]*40,
            "flow_FIN_flag_count":f["fin"],"flow_SYN_flag_count":f["syn"],"flow_RST_flag_count":f["rst"],
            "flow_PSH_flag_count":f["psh"],"flow_ACK_flag_count":f["ack"],
            "fwd_pkts_payload.avg":np.mean(fp),"bwd_pkts_payload.avg":np.mean(bp),
            "fwd_pkts_payload.std":np.std(fp),"bwd_pkts_payload.std":np.std(bp),
            "flow_pkts_payload.avg":np.mean(fp+bp),"flow_pkts_payload.std":np.std(fp+bp),
            "fwd_iat.avg":dur/max(f["fwd"],1)*1000,"bwd_iat.avg":dur/max(f["bwd"],1)*1000,
            "active.avg":dur*1000,"idle.avg":0,
            "_src_ip":f["src"],"_dst_ip":f["dst"],"_pkts":tot,"_duration":dur})
    return records


def extract_csv(path, max_flows):
    df = pd.read_csv(path); print(f"  {len(df):,} rows, {len(df.columns)} columns")
    cols_lower = set(c.strip().lower() for c in df.columns)
    df.columns = [c.strip() for c in df.columns]
    if "flow_pkts_per_sec" not in cols_lower:
        if "flow pkts/s" in cols_lower:
            df.columns = df.columns.str.replace(" ","_").str.replace("/","_per_")
    if args.balanced and "Attack_type" in df.columns:
        types = df["Attack_type"].unique(); per = max(1, max_flows//len(types))
        df = pd.concat([df[df["Attack_type"]==t].sample(n=min(per,len(df[df["Attack_type"]==t])),random_state=42) for t in types]).sample(frac=1,random_state=42).reset_index(drop=True)
        print(f"  Balanced: {len(df)} rows")
    elif len(df) > max_flows:
        df = df.sample(n=max_flows, random_state=42).reset_index(drop=True)
    if "Attack_type" in df.columns:
        print(f"  Types: {df['Attack_type'].value_counts().to_dict()}")
    recs = df.to_dict("records")
    for i,r in enumerate(recs):
        r.setdefault("_src_ip",r.get("src_ip",f"flow_{i}")); r.setdefault("_dst_ip",r.get("dst_ip","?"))
        r.setdefault("_pkts",r.get("fwd_pkts_tot",0)+r.get("bwd_pkts_tot",0))
        r.setdefault("_ground_truth",r.get("Attack_type","?"))
    return recs


class ThreatScorer:
    def __init__(self, path="threat_model.pkl"):
        with open(path,"rb") as f: b=pickle.load(f)
        self.clf,self.scaler,self.feature_names = b["classifier"],b["scaler"],b["feature_names"]
        self.rl_norm = b["rl_norm_params"]; self.n_classes = b.get("n_classes",2); self._c=False
    def score(self, row):
        if not self._c:
            self._c=True; miss=[f for f in self.feature_names if f not in row]
            if miss: print(f"  WARN: {len(miss)} features missing: {miss[:5]}")
            else: print(f"  Features OK: all {len(self.feature_names)} matched")
        feats={f:[float(row.get(f,0.0) if row.get(f,0.0) is not None and not (isinstance(row.get(f,0.0),float) and np.isnan(row.get(f,0.0))) else 0.0)] for f in self.feature_names}
        proba = self.clf.predict_proba(self.scaler.transform(pd.DataFrame(feats)))[0]
        if self.n_classes==2: return float(np.clip(proba[1],0,1))
        return float(np.clip(np.dot(proba,np.linspace(0,1,self.n_classes)),0,1))


class RLAgent:
    def __init__(self, sd=7, use_dqn=None):
        self.sd = sd
        if use_dqn or (use_dqn is None and os.path.exists("dqn_model.pt")):
            from dqn_agent import DQNAgent
            self.a = DQNAgent(sd, NUM_ACTIONS); self.a.load("dqn_model.pt"); self.dqn=True
        elif os.path.exists("q_table.pkl"):
            with open("q_table.pkl","rb") as f: self.Q=pickle.load(f)
            self.bins=[np.linspace(0,1,10)]*sd; self.dqn=False
        else: self.Q={}; self.bins=[np.linspace(0,1,10)]*sd; self.dqn=False
    def act(self, state):
        if self.dqn: return self.a.act(state,training=False), True
        sd=tuple(int(np.clip(np.digitize(state[i],self.bins[i])-1,0,9)) for i in range(min(len(state),self.sd)))
        if sd in self.Q: return int(np.argmax(self.Q[sd])), True
        return (0 if state[0]<0.25 else 1 if state[0]<0.55 else 2), False


def main():
    ext = os.path.splitext(args.input_file)[1].lower()
    print(f"{'='*72}\n  SmartKeyNet Capture Analysis\n{'='*72}\nReading {args.input_file}...")
    flows = extract_pcap(args.input_file, args.max_flows) if ext in (".pcap",".pcapng",".cap") else extract_csv(args.input_file, args.max_flows)
    if not flows: print("No flows."); return
    print(f"  {len(flows)} flows")

    scorer = ThreatScorer()
    sd = 8 if args.use_lstm else 7
    pred = None; th = deque(maxlen=25)
    if args.use_lstm:
        try: from lstm_threat_predictor import ThreatPredictor; pred=ThreatPredictor("lstm_threat.pt")
        except: pass
    use_dqn = True if args.dqn else (False if args.qtable else None)
    agent = RLAgent(sd, use_dqn)

    print(f"\n{'#':>4}  {'Source':>15}  {'Dest':>15}  {'Threat':>6}  {'Decision':>30}  {'Pkts':>5}  Label")
    print("-"*105)
    results=[]; ac={i:0 for i in range(NUM_ACTIONS)}; key_age=0

    for i,flow in enumerate(flows):
        threat = scorer.score(flow); th.append(threat)
        n = scorer.rl_norm
        lat = float(np.clip((np.log1p(float(flow.get("payload_bytes_per_second",0)))-n["latency_min"])/(n["latency_max"]-n["latency_min"]+1e-9),0,1))
        pkt = float(np.clip((float(flow.get("fwd_pkts_tot",0))+float(flow.get("bwd_pkts_tot",0))-n["pktload_min"])/(n["pktload_max"]-n["pktload_min"]+1e-9),0,1))
        qber=float(np.clip(0.04+np.random.normal(0,0.008),0.005,0.12))
        skr=float(np.clip(0.85-2*(qber-0.03)+np.random.normal(0,0.02),0.3,1.0))
        pqc=float(np.clip((0.4+0.45*pkt+0.18*threat)*np.random.normal(1,0.08),0.3,0.9))
        p = pred.predict(list(th)) if pred and len(th)>=5 else None
        state = np.array([threat,qber,skr,pkt,lat,pqc,min(key_age/10,1)],dtype=np.float32)
        if p is not None: state = np.append(state, float(p))
        action, known = agent.act(state); ac[action]+=1
        if action==4: key_age=0
        else: key_age+=1
        src=str(flow.get("_src_ip","?"))[:15]; dst=str(flow.get("_dst_ip","?"))[:15]
        gt=str(flow.get("_ground_truth",""))[:20]; pkts=flow.get("_pkts","?")
        print(f"{i+1:>4}  {src:>15}  {dst:>15}  {threat:>6.3f}  {ACTION_NAMES[action]:>30}  {str(pkts):>5}  {gt}")
        results.append({"threat":threat,"action":action,"ground_truth":gt})

    threats = [r["threat"] for r in results]; total=len(results)
    print(f"\n{'='*72}\n  SUMMARY ({total} flows)\n{'='*72}")
    print(f"  Threat: min={min(threats):.3f} max={max(threats):.3f} avg={np.mean(threats):.3f}")
    lo=sum(1 for t in threats if t<0.25); mi=sum(1 for t in threats if 0.25<=t<0.55)
    hi=sum(1 for t in threats if 0.55<=t<0.75); cr=sum(1 for t in threats if t>=0.75)
    print(f"  Low: {lo} ({lo/total*100:.1f}%)  Mid: {mi} ({mi/total*100:.1f}%)  High: {hi} ({hi/total*100:.1f}%)  Critical: {cr} ({cr/total*100:.1f}%)")
    print(f"\n  Actions:")
    for a,c in ac.items():
        print(f"    {ACTION_NAMES[a]:30s}: {c:4d} ({c/total*100:5.1f}%)  {'█'*int(c/total*50)}")
    print(f"\n  Avg threat per action:")
    for a in range(NUM_ACTIONS):
        at=[r["threat"] for r in results if r["action"]==a]
        if at: print(f"    {ACTION_NAMES[a]:30s}: {np.mean(at):.3f} (n={len(at)})")

    if args.report:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(14,10))
        fig.suptitle(f"SmartKeyNet Analysis — {os.path.basename(args.input_file)}",fontsize=14,fontweight="bold")
        colors=["#1D9E75" if t<0.25 else "#EF9F27" if t<0.55 else "#D85A30" if t<0.75 else "#E24B4A" for t in threats]
        axes[0,0].bar(range(len(threats)),threats,color=colors,width=1,alpha=0.8)
        axes[0,0].set(xlabel="Flow",ylabel="Threat",title="Threat per flow",ylim=(0,1.05))
        for y in [0.25,0.55,0.75]: axes[0,0].axhline(y,color="#888",ls="--",lw=0.8,alpha=0.5)
        labels=[ACTION_NAMES[a] for a in range(NUM_ACTIONS) if ac[a]>0]
        sizes=[ac[a] for a in range(NUM_ACTIONS) if ac[a]>0]
        pcol=["#85B7EB","#AFA9EC","#5DCAA5","#D3D1C7","#F0997B"]
        ucol=[pcol[a] for a in range(NUM_ACTIONS) if ac[a]>0]
        axes[0,1].pie(sizes,labels=labels,colors=ucol,autopct="%1.1f%%",startangle=90,textprops={"fontsize":9})
        axes[0,1].set_title("Action distribution")
        acol={0:"#85B7EB",1:"#AFA9EC",2:"#5DCAA5",3:"#D3D1C7",4:"#F0997B"}
        acts=[r["action"] for r in results]
        for a in range(NUM_ACTIONS):
            m=[i for i,x in enumerate(acts) if x==a]
            if m: axes[1,0].scatter(m,[threats[i] for i in m],c=acol[a],label=ACTION_NAMES[a],s=20,alpha=0.7)
        axes[1,0].set(xlabel="Flow",ylabel="Threat",title="Actions vs threat"); axes[1,0].legend(fontsize=7)
        axes[1,1].hist(threats,bins=30,color="#534AB7",alpha=0.75,edgecolor="white")
        axes[1,1].axvline(np.mean(threats),color="#E24B4A",ls="--",label=f"Mean={np.mean(threats):.3f}")
        axes[1,1].set(xlabel="Threat",ylabel="Count",title="Threat distribution"); axes[1,1].legend()
        plt.tight_layout()
        rp=os.path.splitext(args.input_file)[0]+"_report.png"
        plt.savefig(rp,dpi=150,bbox_inches="tight"); print(f"\n  Report -> {rp}")

if __name__=="__main__": main()
