#!/usr/bin/env python3
"""
Network-namespace testbed for the end-to-end (WAN) experiment.

Creates one Linux network namespace per party, each attached to a shared bridge
via a veth pair, and shapes every party link with `tc`:
  - uplink   (netns eth0 egress)      : rate B                      -> caps what a party sends
  - downlink (host-side veth egress)  : rate B + netem delay RTT/2  -> caps what a party receives
Putting the delay on one direction only makes a round trip incur exactly RTT
(delay on both ends would double it), and capping the downlink at B removes the
single-machine downlink pile-up (each receiver has its own B link).

Parties: slot 0 = server (party id -1), slots 1..N = clients (party id 0..N-1).
IP(slot) = 10.0.0.(slot+2)/24, bridge = 10.0.0.1/24.

Requires root (CAP_NET_ADMIN). Linux only.

Usage:
  sudo python net_setup.py up   --parties 11 --bw 100 --rtt 50
  sudo python net_setup.py check --parties 11
  sudo python net_setup.py down --parties 11
  python net_setup.py up --parties 11 --bw 100 --rtt 50 --dry-run   # print cmds only
"""
import argparse
import os
import subprocess
import sys
import time

BRIDGE = "br-rvfl"

# Kernel input backlog. Every packet a veth delivers goes through the receiving CPU's input
# backlog (net.core.netdev_max_backlog, default 1000 packets, host-wide). When 48 namespaces
# burst into it at once the backlog overflows and the veth silently drops the packet: the
# netns eth0 shows rx_dropped, /proc/net/softnet_stat's dropped column climbs, and TCP sees
# thousands of RTO firings and SYN retransmissions. That turned a 2 s all-to-all exchange
# into 100 s in the sweep. setup() raises the backlog and teardown() restores the old value.
#
# NETEM_LIMIT is the netem queue length in packets. netem is one FIFO per link shared by every
# flow, so its length is the queueing delay a packet can see: at 100 Mbit/s 1000 packets is
# 0.12 s, 100000 packets is 12 s. A long queue does not lose packets but delays ACKs and SYNs
# behind 26 MB of shares until TCP's timers fire anyway (7000 RTO firings with zero drops,
# measured), so the queue must be sized like a router buffer: a couple of bandwidth-delay
# products. "auto" does that per link; a number overrides it.
BACKLOG_PACKETS = int(os.environ.get("RVFL_NETDEV_BACKLOG", "250000"))
NETEM_LIMIT = os.environ.get("RVFL_NETEM_LIMIT", "auto")
NETEM_LIMIT_MIN = 200          # packets; floor for very small bandwidth-delay products

# ARP. The kernel keeps one neighbour table for every namespace on the host, capped by
# net.ipv4.neigh.default.gc_thresh3 (default 1024 entries). 49 namespaces that each talk to
# 48 peers need 2,352 entries, so past the cap "neighbour: arp_cache: neighbor table
# overflow!" fills dmesg, ARP replies are thrown away, and every SYN behind a failed ARP waits
# for the 1 s ARP retry and then its own 1 s SYN timer. Measured on the probe: 1,750 SYN
# retransmissions and 6,600 RTO firings with every drop counter at zero; with the table
# pre-filled, none. setup() raises the thresholds and, unless RVFL_STATIC_ARP=0, writes every
# party's MAC into every other party's table so no ARP exchange happens at all.
NEIGH_THRESH = int(os.environ.get("RVFL_NEIGH_THRESH", "16384"))
STATIC_ARP = os.environ.get("RVFL_STATIC_ARP", "1") == "1"
_NEIGH_SYSCTLS = ["net.ipv4.neigh.default.gc_thresh1", "net.ipv4.neigh.default.gc_thresh2",
                  "net.ipv4.neigh.default.gc_thresh3"]
_SYSCTL_SAVED = {}


def netem_limit(bw_mbps, rtt_ms):
    if NETEM_LIMIT != "auto":
        return int(NETEM_LIMIT)
    bdp_packets = bw_mbps * 1e6 * (rtt_ms / 1000.0) / 8 / 1500
    return max(NETEM_LIMIT_MIN, int(2 * bdp_packets))
_BACKLOG_SYSCTL = "net.core.netdev_max_backlog"
_BACKLOG_SAVED = None
SUBNET = "10.0.0"          # /24; bridge = .1, party slot s = .(s+2)
NS = lambda s: f"rvfl{s}"
VH = lambda s: f"vh{s}"    # host-side veth end
VP = lambda s: f"vp{s}"    # netns-side veth end (renamed to eth0 inside)

DRY = False


def preflight():
    """Fail fast with a clear message if we cannot possibly run the commands."""
    if DRY:
        return
    if os.geteuid() != 0:
        sys.exit('[net_setup] must run as root (CAP_NET_ADMIN).\n'
                 '           Try:  sudo "$(which python)" net_setup.py ...   '
                 '(or: sudo python3 net_setup.py ...)')
    if subprocess.run("which ip", shell=True, capture_output=True).returncode != 0:
        sys.exit("[net_setup] 'ip' (iproute2) not found on root's PATH. Install iproute2.")


def run(cmd, check=True, quiet=False, live=False):
    if DRY:
        print("  " + cmd)
        return 0
    if live:                       # stream stdout/stderr straight to the terminal
        r = subprocess.run(cmd, shell=True)
    else:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0 and check:
        global _FAILS
        _FAILS += 1
        if not quiet and not live:
            print(f"[!] failed: {cmd}\n    {r.stderr.strip()}", file=sys.stderr)
    return r.returncode


_FAILS = 0


def ip_of(slot):
    return f"{SUBNET}.{slot + 2}"


def setup(parties, bw_mbps, rtt_ms):
    preflight()
    global _FAILS
    teardown(parties, announce=False)   # clear any leftover from a previous/crashed run
    _FAILS = 0                           # reset after best-effort cleanup
    half = rtt_ms / 2.0
    limit = netem_limit(bw_mbps, rtt_ms)

    # kernel input backlog and neighbour-table caps (host-wide; see the notes above)
    wanted = {_BACKLOG_SYSCTL: BACKLOG_PACKETS}
    for i, key in enumerate(_NEIGH_SYSCTLS):
        wanted[key] = NEIGH_THRESH * (i + 1) // 2      # thresh1 < thresh2 < thresh3
    for key, val in wanted.items():
        if not DRY:
            r = subprocess.run(f"sysctl -n {key}", shell=True, capture_output=True, text=True)
            _SYSCTL_SAVED[key] = r.stdout.strip() if r.returncode == 0 else None
        run(f"sysctl -q -w {key}={val}")

    # bridge
    run(f"ip link add name {BRIDGE} type bridge", check=False)
    run(f"ip addr add {SUBNET}.1/24 dev {BRIDGE}", check=False)
    run(f"ip link set {BRIDGE} up")

    for s in range(parties):
        ns, vh, vp, ip = NS(s), VH(s), VP(s), ip_of(s)
        # namespace + veth pair
        run(f"ip netns add {ns}")
        run(f"ip link add {vh} type veth peer name {vp}")
        run(f"ip link set {vp} netns {ns}")
        run(f"ip link set {vh} master {BRIDGE}")
        run(f"ip link set {vh} up")
        # inside the namespace
        run(f"ip netns exec {ns} ip link set {vp} name eth0")
        run(f"ip netns exec {ns} ip addr add {ip}/24 dev eth0")
        run(f"ip netns exec {ns} ip link set eth0 up")
        run(f"ip netns exec {ns} ip link set lo up")
        # shaping
        #   uplink   : rate only, on eth0 egress (inside ns)
        #   downlink : rate + delay RTT/2, on host veth egress
        run(f"ip netns exec {ns} tc qdisc add dev eth0 root netem limit {limit} rate {bw_mbps}mbit")
        run(f"tc qdisc add dev {vh} root netem limit {limit} delay {half}ms rate {bw_mbps}mbit")

    if STATIC_ARP:
        _fill_neighbour_tables(parties)

    if _FAILS:
        raise RuntimeError(f"network setup failed with {_FAILS} command error(s) — "
                           f"check the messages above (leftover state / permissions)")
    print(f"[+] up: {parties} parties, {bw_mbps} Mbit/s, RTT {rtt_ms} ms, "
          f"netem limit {limit}, backlog {BACKLOG_PACKETS}, neigh thresh {NEIGH_THRESH}, "
          f"static arp {'on' if STATIC_ARP else 'off'} "
          f"(slot0={ip_of(0)} .. slot{parties-1}={ip_of(parties-1)})")


def _fill_neighbour_tables(parties):
    """Write every party's MAC into every other party's ARP table as a permanent entry, so no
    packet ever waits on an ARP exchange. One `ip -batch` per namespace."""
    macs = {}
    for s in range(parties):
        if DRY:
            macs[s] = "<mac%d>" % s
            continue
        r = subprocess.run(f"ip netns exec {NS(s)} cat /sys/class/net/eth0/address",
                           shell=True, capture_output=True, text=True)
        macs[s] = r.stdout.strip()
    for s in range(parties):
        lines = "".join(f"neigh replace {ip_of(o)} lladdr {macs[o]} dev eth0 nud permanent\n"
                        for o in range(parties) if o != s)
        if DRY:
            print(f"  ip netns exec {NS(s)} ip -batch -   # {parties - 1} neighbour entries")
            continue
        r = subprocess.run(f"ip netns exec {NS(s)} ip -batch -", shell=True, input=lines,
                           capture_output=True, text=True)
        if r.returncode != 0:
            global _FAILS
            _FAILS += 1
            print(f"[!] failed: neighbour entries for {NS(s)}\n    {r.stderr.strip()}", file=sys.stderr)


def teardown(parties, announce=True):
    preflight()
    for s in range(parties):
        run(f"ip netns del {NS(s)}", check=False, quiet=True)
        run(f"ip link del {VH(s)}", check=False, quiet=True)  # usually auto-removed with peer
    run(f"ip link del {BRIDGE}", check=False, quiet=True)
    for key, val in _SYSCTL_SAVED.items():
        if val:
            run(f"sysctl -q -w {key}={val}", check=False, quiet=True)
    if announce:
        print(f"[+] down: removed {parties} namespaces and bridge")


def check(parties):
    """Sanity: ping (RTT) and iperf3 (bandwidth) between slot 0 and slot 1."""
    preflight()
    if parties < 2:
        print("need >=2 parties to check")
        return
    a, b = NS(0), NS(1)
    ipb = ip_of(1)
    print(f"[check] ping {a} -> {ipb}  (expect RTT ~= configured)")
    run(f"ip netns exec {a} ping -c 4 {ipb}", check=False, live=True)

    if subprocess.run("which iperf3", shell=True, capture_output=True).returncode != 0:
        print("[check] iperf3 not installed — skipping bandwidth test "
              "(e.g. apt install iperf3)")
        return
    print(f"[check] iperf3 {a} -> {ipb}  (expect ~= configured bandwidth)")
    srv = subprocess.Popen(f"ip netns exec {b} iperf3 -s -1", shell=True)
    time.sleep(0.5)                                   # let the server bind
    run(f"ip netns exec {a} iperf3 -c {ipb} -t 5", check=False, live=True)
    srv.wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["up", "down", "check"])
    ap.add_argument("--parties", type=int, required=True, help="N+1 (server + N clients)")
    ap.add_argument("--bw", type=float, default=100.0, help="link bandwidth in Mbit/s")
    ap.add_argument("--rtt", type=float, default=50.0, help="round-trip time in ms")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running")
    args = ap.parse_args()
    DRY = args.dry_run

    if args.action == "up":
        setup(args.parties, args.bw, args.rtt)
    elif args.action == "down":
        teardown(args.parties)
    elif args.action == "check":
        check(args.parties)