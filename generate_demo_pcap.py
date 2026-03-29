"""generate_demo_pcap.py — Create demo .pcap with mixed normal + attack traffic."""

import argparse, random, sys
parser = argparse.ArgumentParser()
parser.add_argument("--packets", type=int, default=2000)
parser.add_argument("-o", "--output", default="demo_traffic.pcap")
args = parser.parse_args()

try:
    from scapy.all import IP, TCP, UDP, DNS, DNSQR, Raw, wrpcap, conf; conf.verb=0
except: print("pip install scapy"); sys.exit(1)

print(f"Generating {args.packets} packets...")
pkts=[]; INT=["192.168.1.10","192.168.1.20","192.168.1.30"]
WEB=["93.184.216.34","142.250.80.46"]; ATK="10.0.0.99"; ts=1700000000.0
n=args.packets

for _ in range(int(n*0.4)):
    s,d=random.choice(INT),random.choice(WEB); sp=random.randint(49152,65535)
    pkts.append(IP(src=s,dst=d)/TCP(sport=sp,dport=443,flags="S")); pkts[-1].time=ts
    pkts.append(IP(src=d,dst=s)/TCP(sport=443,dport=sp,flags="SA")); pkts[-1].time=ts+0.01
    pkts.append(IP(src=s,dst=d)/TCP(sport=sp,dport=443,flags="PA")/Raw(b"X"*random.randint(100,1400))); pkts[-1].time=ts+0.02
    ts+=random.uniform(0.05,0.5)
for _ in range(int(n*0.15)):
    pkts.append(IP(src=random.choice(INT),dst="8.8.8.8")/UDP(sport=random.randint(49152,65535),dport=53)/DNS(rd=1,qd=DNSQR(qname="example.com")))
    pkts[-1].time=ts; ts+=random.uniform(0.01,0.1)
for _ in range(int(n*0.15)):
    pkts.append(IP(src=ATK,dst=random.choice(INT))/TCP(sport=random.randint(49152,65535),dport=random.randint(1,10000),flags="S"))
    pkts[-1].time=ts; ts+=random.uniform(0.001,0.01)
for _ in range(int(n*0.15)):
    s=f"{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
    pkts.append(IP(src=s,dst=random.choice(WEB))/TCP(sport=random.randint(1024,65535),dport=80,flags="S"))
    pkts[-1].time=ts; ts+=random.uniform(0.0005,0.005)
for _ in range(int(n*0.15)):
    pkts.append(IP(src=random.choice(INT),dst="45.33.32.156")/TCP(sport=random.randint(49152,65535),dport=443,flags="PA")/Raw(b"\x00"*random.randint(1000,1460)))
    pkts[-1].time=ts; ts+=random.uniform(0.01,0.05)

pkts.sort(key=lambda p:p.time); wrpcap(args.output, pkts)
print(f"Saved {len(pkts):,} packets -> {args.output}")
print(f"Analyze: python analyze_capture.py {args.output} --report")
