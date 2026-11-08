/*
 * FRLOR forwarding agent for NS-2.34  (candidate lists are computed offline by the python package)
 *
 * Opportunistic forwarding, timer-based coordination (paper Sec. 4, Eq. 7):
 *   - a node broadcasts the packet with its priority-ordered candidate list in the header;
 *   - candidate i (0-based) that receives it waits  i * T_default  (T_i = (i-1)*T_default, i 1-based);
 *   - if, while waiting, it overhears the same packet being forwarded by another node, it cancels (suppression);
 *   - the sender uses overhearing of a forward (or a destination ACK) as implicit acknowledgement and
 *     retransmits after  |candidates|*T_default + ack_margin  if none is heard (up to max_retry_).
 */
#ifndef ns_frlor_agent_h
#define ns_frlor_agent_h

#include <map>
#include <vector>
#include <stdio.h>
#include "agent.h"
#include "packet.h"
#include "ip.h"
#include "timer-handler.h"

#define FRLOR_MAXC 8
#define FRLOR_DATA 0
#define FRLOR_ACK  1

struct hdr_frlor {
    int    type_;                 // FRLOR_DATA / FRLOR_ACK
    int    src_;                  // originating node
    int    dst_;                  // final destination
    int    seq_;                  // sequence number at the originator
    int    ncand_;                // number of valid entries in cand_
    int    cand_[FRLOR_MAXC];     // candidate list, highest priority first
    double t0_;                   // origination time (for end-to-end delay)
    int    hops_;

    static int offset_;
    inline static int& offset() { return offset_; }
    inline static hdr_frlor* access(const Packet* p) { return (hdr_frlor*) p->access(offset_); }
};

class FRLORAgent;

class FwdTimer : public TimerHandler {
public:
    FwdTimer(FRLORAgent* a, long long k) : TimerHandler(), a_(a), key_(k) {}
    void expire(Event*);
private:
    FRLORAgent* a_; long long key_;
};

class AckTimer : public TimerHandler {
public:
    AckTimer(FRLORAgent* a, long long k) : TimerHandler(), a_(a), key_(k) {}
    void expire(Event*);
private:
    FRLORAgent* a_; long long key_;
};

struct PktState {
    PktState() : heard(false), from(-1), forwarded(false), suppressed(false), acked(false),
                 retries(0), src(-1), dst(-1), seq(0), hops(0), t0(0.0), fwd(0), ack(0) {}
    bool heard, forwarded, suppressed, acked;
    int  from, retries, src, dst, seq, hops;
    double t0;
    FwdTimer* fwd;
    AckTimer* ack;
};

class FRLORAgent : public Agent {
public:
    FRLORAgent();
    int  command(int argc, const char* const* argv);
    void recv(Packet*, Handler*);
    void on_fwd_timeout(long long key);
    void on_ack_timeout(long long key);

private:
    void originate();
    void transmit(PktState& s, int type);
    void start_ack_timer(PktState& s, long long key);
    void overheard(PktState& s, long long key);
    void load_candidates(const char* file);

    std::vector<int> cand_;                 // my candidate list (from the offline file)
    std::map<long long, PktState> st_;      // per-packet state, key = (src<<32)|seq
    int    dest_, seq_;
    int    max_retry_, payload_;            // bound OTcl variables
    double tdefault_, ack_margin_;
};

#endif
