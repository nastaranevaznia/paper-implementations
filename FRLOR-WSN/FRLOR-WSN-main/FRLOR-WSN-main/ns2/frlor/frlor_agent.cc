#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include "frlor_agent.h"
#include "node.h"
#include "energy-model.h"

int hdr_frlor::offset_;

static class FRLORHeaderClass : public PacketHeaderClass {
public:
    FRLORHeaderClass() : PacketHeaderClass("PacketHeader/FRLOR", sizeof(hdr_frlor)) {
        bind_offset(&hdr_frlor::offset_);
    }
} class_frlorhdr;

static class FRLORAgentClass : public TclClass {
public:
    FRLORAgentClass() : TclClass("Agent/FRLOR") {}
    TclObject* create(int, const char* const*) { return new FRLORAgent(); }
} class_frlor;

/* ---- shared log file (one line per event, parsed by frlor/analyze_ns2.py) ---- */
static FILE* g_log = 0;
static void L(const char* fmt, ...) {
    if (!g_log) return;
    va_list ap; va_start(ap, fmt); vfprintf(g_log, fmt, ap); va_end(ap);
    fflush(g_log);
}
static inline long long mk(int src, int seq) { return (((long long) src) << 32) | (unsigned int) seq; }

void FwdTimer::expire(Event*) { a_->on_fwd_timeout(key_); }
void AckTimer::expire(Event*) { a_->on_ack_timeout(key_); }

FRLORAgent::FRLORAgent() : Agent(PT_FRLOR), dest_(-1), seq_(0), max_retry_(7), payload_(512),
                           tdefault_(0.05), ack_margin_(0.15) {
    bind("T_default_",  &tdefault_);
    bind("ack_margin_", &ack_margin_);
    bind("max_retry_",  &max_retry_);
    bind("payload_",    &payload_);
}

int FRLORAgent::command(int argc, const char* const* argv) {
    if (argc == 2) {
        if (!strcmp(argv[1], "send")) { originate(); return TCL_OK; }
        if (!strcmp(argv[1], "dump-energy")) {
            Node* n = Node::get_node_by_address(addr());
            double e = -1.0;
            if (n && n->energy_model()) e = n->energy_model()->energy();
            L("ENERGY %d %.6f\n", (int) addr(), e);
            return TCL_OK;
        }
        if (!strcmp(argv[1], "close-log")) { if (g_log) { fclose(g_log); g_log = 0; } return TCL_OK; }
    } else if (argc == 3) {
        if (!strcmp(argv[1], "set-dest")) { dest_ = atoi(argv[2]); return TCL_OK; }
        if (!strcmp(argv[1], "load-candidates")) { load_candidates(argv[2]); return TCL_OK; }
        if (!strcmp(argv[1], "log")) { if (!g_log) g_log = fopen(argv[2], "w"); return TCL_OK; }
    }
    return Agent::command(argc, argv);
}

void FRLORAgent::load_candidates(const char* file) {
    FILE* f = fopen(file, "r");
    if (!f) { fprintf(stderr, "FRLOR: cannot open %s\n", file); exit(1); }
    char line[1024];
    cand_.clear();
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char* tok = strtok(line, " \t\r\n");
        if (!tok || atoi(tok) != (int) addr()) continue;
        while ((tok = strtok(NULL, " \t\r\n")) != NULL && (int) cand_.size() < FRLOR_MAXC)
            cand_.push_back(atoi(tok));
        break;
    }
    fclose(f);
}

void FRLORAgent::transmit(PktState& s, int type) {
    Packet* p = allocpkt();
    hdr_cmn* ch = hdr_cmn::access(p);
    hdr_ip*  ih = hdr_ip::access(p);
    hdr_frlor* fh = hdr_frlor::access(p);

    ch->ptype()     = PT_FRLOR;
    ch->size()      = (type == FRLOR_DATA) ? payload_ + 4 * FRLOR_MAXC : 40;
    ch->addr_type() = NS_AF_NONE;
    ch->next_hop()  = IP_BROADCAST;
    ch->direction() = hdr_cmn::DOWN;
    ih->saddr() = addr();  ih->sport() = port();
    ih->daddr() = IP_BROADCAST;  ih->dport() = port();
    ih->ttl_ = 32;

    fh->type_ = type;  fh->src_ = s.src;  fh->dst_ = s.dst;  fh->seq_ = s.seq;
    fh->t0_ = s.t0;    fh->hops_ = s.hops;
    fh->ncand_ = (type == FRLOR_DATA) ? (int) cand_.size() : 0;
    for (int i = 0; i < fh->ncand_; i++) fh->cand_[i] = cand_[i];

    if (type == FRLOR_DATA)
        L("TX %.6f %d %d %d %d %d\n", Scheduler::instance().clock(), (int) addr(), s.src, s.seq, s.retries, s.hops);
    send(p, 0);
}

void FRLORAgent::start_ack_timer(PktState& s, long long key) {
    if (!s.ack) s.ack = new AckTimer(this, key);
    s.ack->resched(cand_.size() * tdefault_ + ack_margin_);
}

void FRLORAgent::originate() {
    double now = Scheduler::instance().clock();
    long long key = mk(addr(), ++seq_);
    PktState& s = st_[key];
    s.heard = s.forwarded = true;  s.src = addr();  s.dst = dest_;  s.seq = seq_;  s.t0 = now;  s.hops = 0;
    L("SRC %.6f %d %d\n", now, (int) addr(), seq_);
    if (cand_.empty()) { L("DROP %.6f %d %d %d void\n", now, (int) addr(), s.src, s.seq); return; }
    transmit(s, FRLOR_DATA);
    start_ack_timer(s, key);
}

void FRLORAgent::overheard(PktState& s, long long) {
    if (s.fwd && s.fwd->status() == TIMER_PENDING) { s.fwd->cancel(); s.suppressed = true; }
    if (s.ack && s.ack->status() == TIMER_PENDING) { s.ack->cancel(); s.acked = true; }
}

void FRLORAgent::recv(Packet* p, Handler*) {
    hdr_frlor* fh = hdr_frlor::access(p);
    int sender = hdr_ip::access(p)->saddr();
    double now = Scheduler::instance().clock();
    if (sender == (int) addr()) { Packet::free(p); return; }

    long long key = mk(fh->src_, fh->seq_);
    PktState& s = st_[key];

    if (fh->type_ == FRLOR_ACK) {                 // destination ACK: packet is delivered, stop everything
        s.heard = true;
        overheard(s, key);
        Packet::free(p);
        return;
    }

    if ((int) addr() == fh->dst_) {               // I am the destination
        if (!s.heard) {
            s.heard = true;
            L("RX_DEST %.6f %d %d %.6f %d\n", now, fh->src_, fh->seq_, now - fh->t0_, fh->hops_);
        }
        s.src = fh->src_;  s.dst = fh->dst_;  s.seq = fh->seq_;  s.t0 = fh->t0_;  s.hops = fh->hops_;
        transmit(s, FRLOR_ACK);                   // (re-)acknowledge; not counted as a data transmission
        Packet::free(p);
        return;
    }

    if (s.heard) {                                // duplicate
        if (sender != s.from) overheard(s, key);  // someone else forwarded -> implicit ACK / suppression
        Packet::free(p);
        return;
    }

    s.heard = true;  s.from = sender;
    s.src = fh->src_;  s.dst = fh->dst_;  s.seq = fh->seq_;  s.t0 = fh->t0_;  s.hops = fh->hops_;
    int idx = -1;
    for (int i = 0; i < fh->ncand_; i++) if (fh->cand_[i] == (int) addr()) { idx = i; break; }
    if (idx >= 0) {                               // I am a candidate: wait idx * T_default
        if (!s.fwd) s.fwd = new FwdTimer(this, key);
        s.fwd->resched(idx * tdefault_);
    }
    Packet::free(p);
}

void FRLORAgent::on_fwd_timeout(long long key) {
    PktState& s = st_[key];
    if (s.suppressed || s.forwarded) return;
    s.forwarded = true;
    s.hops++;
    if (cand_.empty()) { L("DROP %.6f %d %d %d void\n", Scheduler::instance().clock(), (int) addr(), s.src, s.seq); return; }
    s.retries = 0;
    transmit(s, FRLOR_DATA);
    start_ack_timer(s, key);
}

void FRLORAgent::on_ack_timeout(long long key) {
    PktState& s = st_[key];
    if (s.acked) return;
    if (s.retries < max_retry_) {
        s.retries++;
        transmit(s, FRLOR_DATA);
        start_ack_timer(s, key);
    } else {
        L("DROP %.6f %d %d %d retry\n", Scheduler::instance().clock(), (int) addr(), s.src, s.seq);
    }
}
