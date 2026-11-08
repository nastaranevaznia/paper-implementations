#!/bin/sh
# Install the FRLOR agent into an NS-2.34 source tree.   usage:  sh install_ns2.sh /path/to/ns-allinone-2.34/ns-2.34
# Idempotent; every modified file is backed up once as <file>.frlor.orig.  Afterwards: cd $NS2 && make
set -e
NS2="$1"; [ -d "$NS2/common" ] || { echo "usage: $0 <ns-2.34 source dir>"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
bak() { [ -f "$1.frlor.orig" ] || cp "$1" "$1.frlor.orig"; }

mkdir -p "$NS2/frlor" && cp "$HERE"/frlor/frlor_agent.h "$HERE"/frlor/frlor_agent.cc "$NS2/frlor/"

# 1. new packet type -------------------------------------------------------------------------------
P="$NS2/common/packet.h"; bak "$P"
grep -q PT_FRLOR "$P" || {
  sed -i 's|^\([[:space:]]*\)PT_NTYPE|\1PT_FRLOR,\n\1PT_NTYPE|' "$P"
  sed -i 's|^\([[:space:]]*\)name_\[PT_NTYPE\]|\1name_[PT_FRLOR]= "FRLOR";\n\1name_[PT_NTYPE]|' "$P"; }
grep -q 'PT_FRLOR,' "$P" && grep -q 'name_\[PT_FRLOR\]' "$P" || echo "!! packet.h not patched automatically: add PT_FRLOR before PT_NTYPE and name_[PT_FRLOR]=\"FRLOR\" by hand"

# 2. packet header registration (OTcl) -------------------------------------------------------------
T="$NS2/tcl/lib/ns-packet.tcl"; bak "$T"
grep -q '^[[:space:]]*FRLOR' "$T" || sed -i '0,/^[[:space:]]*AODV[[:space:]]*$/s//\tAODV\n\tFRLOR/' "$T"
grep -q '^[[:space:]]*FRLOR' "$T" || echo "!! ns-packet.tcl not patched: add a line 'FRLOR' next to 'AODV' in the foreach prot {...} list"

# 3. default values --------------------------------------------------------------------------------
D="$NS2/tcl/lib/ns-default.tcl"; bak "$D"
grep -q 'Agent/FRLOR' "$D" || cat >> "$D" <<'EOT'
Agent/FRLOR set T_default_ 0.05
Agent/FRLOR set ack_margin_ 0.15
Agent/FRLOR set max_retry_ 7
Agent/FRLOR set payload_ 512
EOT

# 4. build system ----------------------------------------------------------------------------------
for M in "$NS2/Makefile.in" "$NS2/Makefile"; do
  [ -f "$M" ] || continue; bak "$M"
  grep -q 'frlor/frlor_agent.o' "$M" || sed -i 's|^OBJ_CC = |OBJ_CC = frlor/frlor_agent.o |' "$M"
  grep -q 'frlor/frlor_agent.o' "$M" || echo "!! $M not patched: add frlor/frlor_agent.o to OBJ_CC"
done

# 5. silence AODV HELLO beacons (AODV is only a port-demux shim here) ------------------------------
A="$NS2/aodv/aodv.cc"; bak "$A"
sed -i 's|^\([[:space:]]*\)htimer.handle((Event\*) 0);|\1/* FRLOR: HELLO disabled */|; s|^\([[:space:]]*\)ntimer.handle((Event\*) 0);|\1/* FRLOR: neighbour timer disabled */|' "$A"
grep -q 'FRLOR: HELLO disabled' "$A" || echo "!! aodv.cc: could not disable HELLO; comment out htimer/ntimer.handle in AODV::command(\"start\")"

echo "done. now:  cd $NS2 && make   (then ./ns ../path/to/frlor_sim.tcl ...)"
