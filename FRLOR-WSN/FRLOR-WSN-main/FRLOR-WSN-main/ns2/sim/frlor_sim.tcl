# ---------------------------------------------------------------------------------------------
# FRLOR / POR / fuzzy-only opportunistic routing in NS-2.34   (static WSN, shadowing channel)
#
#   ns frlor_sim.tcl <topology.txt> <candidates.txt> <log.txt> <n_nodes> <src> <dst> \
#                    <n_packets> <interval_s> <seed> [initial_energy_J]
#
# topology.txt / candidates.txt are produced by `python -m frlor.pipeline` (same node ids).
# Radio constants = paper Table 1; shadowing beta = 2.7, sigma = 6 dB (Sec. 3.3).
# ---------------------------------------------------------------------------------------------
if {$argc < 9} {
    puts "usage: ns frlor_sim.tcl topo cand log n_nodes src dst n_packets interval seed \[init_energy\]"
    exit 1
}
set topofile [lindex $argv 0]
set candfile [lindex $argv 1]
set logfile  [lindex $argv 2]
set nn       [lindex $argv 3]
set src      [lindex $argv 4]
set dst      [lindex $argv 5]
set npkts    [lindex $argv 6]
set interval [lindex $argv 7]
set seed     [lindex $argv 8]
set initE    100.0
if {$argc > 9} { set initE [lindex $argv 9] }
set width 1200
set height 1200
set port 200                              ;# 255 is reserved for the routing agent (AODV, used as port-demux shim)
set tstart 10.0                           ;# let the stack settle before the first packet
set tend [expr $tstart + $npkts * $interval + 10.0]

# --- protocol parameters (paper Sec. 4 / Eq. 7) --------------------------------------------------
Agent/FRLOR set T_default_  0.05          ;# 50 ms
Agent/FRLOR set ack_margin_ 0.15
Agent/FRLOR set max_retry_  7
Agent/FRLOR set payload_    512

# --- radio / channel (Table 1) ---------------------------------------------------------------------
Propagation/Shadowing set pathlossExp_ 2.7
Propagation/Shadowing set std_db_      6.0
Propagation/Shadowing set dist0_       1.0
Propagation/Shadowing set seed_        $seed
Phy/WirelessPhy set Pt_       0.28183815
Phy/WirelessPhy set RXThresh_ 3.652e-10
Phy/WirelessPhy set freq_     914e6
Phy/WirelessPhy set L_        1.0
Antenna/OmniAntenna set Gt_ 1.0
Antenna/OmniAntenna set Gr_ 1.0

set ns [new Simulator]
set topo [new Topography]
$topo load_flatgrid $width $height
create-god $nn

# AODV is used ONLY as the NS-2 shim that hands broadcast data packets to the port demux
# (its HELLO timers are disabled by ns2/install_ns2.sh); all forwarding decisions are made by Agent/FRLOR.
$ns node-config -adhocRouting AODV \
    -llType LL -macType Mac/802_11 \
    -ifqType Queue/DropTail/PriQueue -ifqLen 50 \
    -antType Antenna/OmniAntenna \
    -propType Propagation/Shadowing \
    -phyType Phy/WirelessPhy \
    -channelType Channel/WirelessChannel \
    -topoInstance $topo \
    -agentTrace OFF -routerTrace OFF -macTrace OFF -movementTrace OFF \
    -energyModel EnergyModel -initialEnergy $initE \
    -txPower 0.660 -rxPower 0.395 -idlePower 0.0 -sleepPower 0.0

# --- read topology ------------------------------------------------------------------------------------
set fp [open $topofile r]
while {[gets $fp line] >= 0} {
    if {[string index $line 0] == "#" || [string trim $line] == ""} { continue }
    set id [lindex $line 0]
    set px($id) [lindex $line 1]
    set py($id) [lindex $line 2]
}
close $fp

for {set i 0} {$i < $nn} {incr i} {
    set node($i) [$ns node]
    $node($i) random-motion 0
    $node($i) set X_ $px($i)
    $node($i) set Y_ $py($i)
    $node($i) set Z_ 0.0
    $ns initial_node_pos $node($i) 10

    set agent($i) [new Agent/FRLOR]
    $node($i) attach $agent($i) $port
    $agent($i) set dst_port_ $port
    $agent($i) set-dest $dst
    $agent($i) load-candidates $candfile
    $agent($i) log $logfile
}

for {set k 0} {$k < $npkts} {incr k} {
    $ns at [expr $tstart + $k * $interval] "$agent($src) send"
}

proc finish {} {
    global ns nn agent
    for {set i 0} {$i < $nn} {incr i} { $agent($i) dump-energy }
    $agent(0) close-log
    $ns flush-trace
    puts "FRLOR simulation finished"
    exit 0
}
$ns at $tend "finish"
$ns run
