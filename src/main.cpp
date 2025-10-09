// main.cpp (refactored, DT-only, readable)
// ============================================================================
//  센터라인 & 폭 계산 (오픈/폐루프) + 최소 곡률 레이싱라인
//  - cfg::Config 로 동작 제어
//  - DT(Bowyer–Watson)만 사용 (CDT/클리핑/품질필터 제거)
//  - MST 기반 중점 순서화 + OPEN/폐곡 "방향 & 시작점 정합"
//  - inner/outer를 중점 진행방향과 동일하게 정렬
//  - 스플라인 균일 재샘플 → 폭/기하량 계산 → 최소곡률 레이싱라인 최적화
//  - 각 단계 CSV 덤프(디버깅 친화)
// ============================================================================

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

using std::pair;
using std::string;
using std::vector;
using std::cerr;

// ================================ Config ==================================
namespace cfg {
struct Config {
    // Anchors
    double current_pos_x = 4.994457593, current_pos_y = -0.108866966; // (open용)
    double current_heading_rad = 0.046698693;
    double start_anchor_x = 0.0, start_anchor_y = 0.0;   // (closed 시작점 회전)
    double start_heading_rad = 0.0;                      // (closed 방향 정합)

    bool  is_closed_track = true;        // 폐/비폐 모드
    bool  emit_closed_duplicate = true;  // 폐루프 결과 마지막=첫점 복제
    bool  verbose = true;                   // 로그 출력

    // 선택: 폐루프를 무조건 CCW로 강제할지 (기본 false)
    bool  force_closed_ccw = false;

    // 샘플링
    bool   add_micro_jitter = true;
    double jitter_eps = 1e-9;

    int    samples = 300;
    bool   use_dynamic_samples = true;
    double sample_factor_n = 1.0;
    int    samples_min = 10;
    int    samples_max = 1000;

    int    knn_k = 4; // MST k-NN

    // 경계 엣지 길이 필터
    bool   enable_boundary_len_filter = true;
    double boundary_edge_len_scale = 2.0;
    double boundary_edge_abs_max   = 6.0;

    // 코리도/차량
    double veh_width_m = 1.0;
    double safety_margin_m = 0.05;

    // 최소곡률 최적화
    double lambda_smooth = 1e-3;
    int    max_outer_iters = 16;
    int    max_inner_iters = 150;
    double step_init = 0.6;
    double step_min  = 1e-6;
    double armijo_c  = 1e-5;

    double kappa_eps = 1e-6;
    double v_cap_mps = 20.0;

    // --- dynamics/aero ---
    double mass_kg      = 255.0;
    double Cd           = 0.30;
    double A_front_m2   = 1.00;    // or set K_drag directly
    double rho_air      = 1.225;
    double c_rr         = 0.015;
    double P_max_W      = 80000.0; // 60kW~80kW 사이 조정 가능

    // friction circle (lat/long tradeoff)
    double mu           = 1.1;    // dry asphalt with slicks-ish
    double a_total_max  = mu * 9.81;  // ≈ 11.5
    double a_lat_max = a_total_max;
    double a_long_acc_cap   = 5.0;
    double a_long_brake_cap = 5.0;

    double w_time_gain = 8.0;      // [-] min-time 가중치 강도 (lat-리미트 구간에 가중)
    int    max_vpass_iters = 6;    // [-] v(s) 전/후진 패스 반복 횟수

    // === DEBUG ===
    bool   debug_dump = true;        // 디버그 덤프 활성화
    double debug_offset_warn_m = 0.05; // 센터라인 대비 편차 경고 기준 (m)

    double time_gamma_power = 2.0;    // γ 가중 승수 지수(2~6 추천)
    bool   time_weight_use_inv_v = true; // 원하면 1/v 가중 추가
    double inv_v_gain = 0.8;
    bool   use_total_ge_lat = true;   // 총가속 >= 횡가속 한계 보장
};
inline Config& get(){ static Config C; return C; }
} // namespace cfg

// =============================== Geometry =================================
namespace geom {
struct Vec2 { double x=0, y=0; };
inline Vec2 operator+(const Vec2&a,const Vec2&b){ return {a.x+b.x,a.y+b.y}; }
inline Vec2 operator-(const Vec2&a,const Vec2&b){ return {a.x-b.x,a.y-b.y}; }
inline Vec2 operator*(const Vec2&a,double s){ return {a.x*s,a.y*s}; }
inline double dot(const Vec2&a,const Vec2&b){ return a.x*b.x + a.y*b.y; }
inline double norm2(const Vec2&a){ return dot(a,a); }
inline double norm(const Vec2&a){ return std::sqrt(norm2(a)); }
inline Vec2 normalize(const Vec2&v,double eps=1e-12){ double n=norm(v); return (n<eps)?Vec2{0,0}:Vec2{v.x/n,v.y/n}; }
inline bool almostEq(const Vec2&a,const Vec2&b,double e=1e-12){ return (std::fabs(a.x-b.x)<=e && std::fabs(a.y-b.y)<=e); }

// robust predicates
inline double orient2d_filt(const Vec2&a,const Vec2&b,const Vec2&c){
    double det=(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
    double absa=std::fabs(b.x-a.x)+std::fabs(b.y-a.y);
    double absb=std::fabs(c.x-a.x)+std::fabs(c.y-a.y);
    double err=(absa*absb)*std::numeric_limits<double>::epsilon()*4.0;
    if (std::fabs(det)>err) return det;
    long double adx=(long double)b.x - (long double)a.x;
    long double ady=(long double)b.y - (long double)a.y;
    long double bdx=(long double)c.x - (long double)a.x;
    long double bdy=(long double)c.y - (long double)a.y;
    long double detl=adx*bdy - ady*bdx;
    return (double)detl;
}
inline double incircle_filt(const Vec2&a,const Vec2&b,const Vec2&c,const Vec2&d){
    double adx=a.x-d.x, ady=a.y-d.y;
    double bdx=b.x-d.x, bdy=b.y-d.y;
    double cdx=c.x-d.x, cdy=c.y-d.y;
    double ad=adx*adx+ady*ady, bd=bdx*bdx+bdy*bdy, cd=cdx*cdx+cdy*cdy;
    double det=adx*(bdy*cd - bd*cdy) - ady*(bdx*cd - bd*cdx) + ad*(bdx*cdy - bdy*cdx);
    double mags=(std::fabs(adx)+std::fabs(ady))*(std::fabs(bdx)+std::fabs(bdy))*(std::fabs(cdx)+std::fabs(cdy));
    double err=mags*std::numeric_limits<double>::epsilon()*16.0;
    if (std::fabs(det)>err) return det;
    long double AX=a.x, AY=a.y, BX=b.x, BY=b.y, CX=c.x, CY=c.y, DX=d.x, DY=d.y;
    long double adxl=AX-DX, adyl=AY-DY, bdxl=BX-DX, bdyl=BY-DY, cdxl=CX-DX, cdyl=CY-DY;
    long double adl=adxl*adxl+adyl*adyl, bdl=bdxl*bdxl+bdyl*bdyl, cdl=cdxl*cdxl+cdyl*cdyl;
    long double detl=adxl*(bdyl*cdl - bdl*cdyl) - adyl*(bdxl*cdl - bdl*cdxl) + adl*(bdxl*cdyl - bdyl*cdxl);
    return (double)detl;
}
inline bool ccw(const Vec2&A,const Vec2&B,const Vec2&C){ return orient2d_filt(A,B,C) > 0; }
inline double polygonSignedArea(const std::vector<Vec2>&P){
    if (P.size()<3) return 0.0;
    long double s=0; size_t n=P.size();
    for(size_t i=0;i<n;i++){ size_t j=(i+1)%n; s += (long double)P[i].x*P[j].y - (long double)P[j].x*P[i].y; }
    return (double)(0.5L*s);
}
} // namespace geom

// ============================== Delaunay (DT) =============================
namespace delaunay {
using geom::Vec2;

struct Tri{ int a,b,c; }; // CCW

static vector<Tri> bowyerWatson(const vector<Vec2>& pts){
    auto &C = cfg::get();
    vector<Vec2> P = pts;
    if (C.add_micro_jitter){
        std::mt19937_64 rng(1234567);
        std::uniform_real_distribution<double> U(-C.jitter_eps, C.jitter_eps);
        for (auto &p: P){ p.x += U(rng); p.y += U(rng); }
    }
    // super-triangle
    geom::Vec2 lo{+1e300,+1e300}, hi{-1e300,-1e300};
    for (auto&p: P){ lo.x=std::min(lo.x,p.x); lo.y=std::min(lo.y,p.y); hi.x=std::max(hi.x,p.x); hi.y=std::max(hi.y,p.y); }
    geom::Vec2 c={(lo.x+hi.x)*0.5,(lo.y+hi.y)*0.5};
    double d = std::max(hi.x-lo.x, hi.y-lo.y)*1000.0 + 1.0;
    int n0 = (int)P.size();
    P.push_back({c.x-2*d, c.y-d});
    P.push_back({c.x+2*d, c.y-d});
    P.push_back({c.x,     c.y+2*d});
    int s1=n0, s2=n0+1, s3=n0+2;

    vector<Tri> T; T.push_back({s1,s2,s3});
    for (int ip=0; ip<n0; ++ip){
        const Vec2 &p = P[ip];
        vector<int> bad; bad.reserve(T.size()/3);
        for (int t=0; t<(int)T.size(); ++t){
            auto &tr=T[t];
            if (!geom::ccw(P[tr.a],P[tr.b],P[tr.c])) std::swap(tr.b,tr.c);
            if (geom::incircle_filt(P[tr.a],P[tr.b],P[tr.c],p) > 0) bad.push_back(t);
        }
        struct E{int u,v;};
        vector<E> poly;
        auto addE=[&](int u,int v){
            for (auto it=poly.begin(); it!=poly.end(); ++it){
                if (it->u==v && it->v==u){ poly.erase(it); return; }
            }
            poly.push_back({u,v});
        };
        vector<char> del(T.size(),0);
        for (int id: bad){
            del[id]=1; auto tr=T[id];
            addE(tr.a,tr.b); addE(tr.b,tr.c); addE(tr.c,tr.a);
        }
        vector<Tri> keep; keep.reserve(T.size());
        for (int i=0;i<(int)T.size();++i) if(!del[i]) keep.push_back(T[i]);
        T.swap(keep);
        for (auto &e: poly){
            Tri nt{e.u,e.v,ip};
            if (!geom::ccw(P[nt.a],P[nt.b],P[nt.c])) std::swap(nt.b,nt.c);
            T.push_back(nt);
        }
    }
    vector<Tri> out; out.reserve(T.size());
    for (auto &tr: T){ if (tr.a<n0 && tr.b<n0 && tr.c<n0) out.push_back(tr); }
    return out;
}

struct EdgeKey{ int u,v; EdgeKey(){} EdgeKey(int a,int b){u=std::min(a,b); v=std::max(a,b);} bool operator==(const EdgeKey&o)const{return u==o.u&&v==o.v;} };
struct EdgeKeyHash{ size_t operator()(const EdgeKey&k)const{ return ((uint64_t)k.u<<32) ^ (uint64_t)k.v; } };
struct EdgeRef{ int tri; int a,b; };

inline void buildEdgeMap(const vector<Tri>&T, std::unordered_map<EdgeKey, vector<EdgeRef>, EdgeKeyHash>&M){
    M.clear(); M.reserve(T.size()*2);
    for (int t=0;t<(int)T.size();++t){
        const Tri &tr=T[t]; int A[3]={tr.a,tr.b,tr.c};
        for(int i=0;i<3;i++){ int u=A[i], v=A[(i+1)%3]; M[EdgeKey(u,v)].push_back({t,u,v}); }
    }
}
} // namespace delaunay

// ============================== Edge helpers ==============================
namespace edges {
using geom::Vec2;

inline vector<pair<Vec2,Vec2>> ringEdges(const vector<Vec2>&R){
    vector<pair<Vec2,Vec2>> E; int n=(int)R.size(); E.reserve(n);
    for(int i=0;i<n;i++){ int j=(i+1)%n; E.push_back({R[i], R[j]}); }
    return E;
}
inline vector<pair<Vec2,Vec2>> polylineEdges(const vector<Vec2>&R){
    vector<pair<Vec2,Vec2>> E; int n=(int)R.size(); if(n<2) return E; E.reserve(n-1);
    for(int i=0;i+1<n;i++) E.push_back({R[i], R[i+1]});
    return E;
}
} // namespace edges

// =============================== IO Utils =================================
namespace io {
using geom::Vec2;

inline vector<Vec2> loadCSV_XY(const string& path){
    vector<Vec2> pts; std::ifstream fin(path);
    if(!fin){ std::cerr << "[ERR] cannot open: " << path << "\n"; return pts; }
    string line;
    while(std::getline(fin,line)){
        if(line.empty()) continue;
        for(char &ch: line) if(ch==';' || ch=='\t') ch=' ';
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        double x,y; if(iss>>x>>y) pts.push_back({x,y});
    }
    return pts;
}
inline bool saveCSV_pointsXY(const string& path, const vector<Vec2>& pts){
    std::ofstream fo(path); if(!fo){ std::cerr<<"[ERR] write failed: "<<path<<"\n"; return false; }
    fo.setf(std::ios::fixed); fo.precision(9);
    for(auto&p: pts) fo<<p.x<<","<<p.y<<"\n"; return true;
}
inline bool saveCSV_pointsLabeled(const string& path, const vector<Vec2>& pts, const vector<int>& label){
    std::ofstream fo(path); if(!fo){ std::cerr<<"[ERR] write failed: "<<path<<"\n"; return false; }
    fo<<"id,x,y,label\n"; fo.setf(std::ios::fixed); fo.precision(9);
    for(size_t i=0;i<pts.size();++i) fo<<i<<","<<pts[i].x<<","<<pts[i].y<<","<<label[i]<<"\n"; return true;
}
inline bool saveCSV_edgesIdx(const string& path, const vector<pair<int,int>>&E){
    std::ofstream fo(path); if(!fo){ std::cerr<<"[ERR] write failed: "<<path<<"\n"; return false; }
    for(auto&e:E) fo<<e.first<<","<<e.second<<"\n"; return true;
}
inline bool saveCSV_trisIdx(const string& path, const vector<delaunay::Tri>&T){
    std::ofstream fo(path); if(!fo){ std::cerr<<"[ERR] write failed: "<<path<<"\n"; return false; }
    for(auto&t:T) fo<<t.a<<","<<t.b<<","<<t.c<<"\n"; return true;
}
inline string dropExt(const string& s){ size_t p=s.find_last_of('.'); return (p==string::npos)?s:s.substr(0,p); }
} // namespace io

// ============================== Centerline =================================
namespace centerline {
using geom::Vec2; using delaunay::Tri;

struct BoundaryEdgeInfo{
    int u,v; double len; bool is_hull; Vec2 mid;
};

inline vector<BoundaryEdgeInfo> labelBoundaryEdges_with_len(const vector<Vec2>&all,
                                                            const vector<Tri>&T,
                                                            const vector<int>&labels){
    using namespace delaunay;
    std::unordered_map<EdgeKey, vector<EdgeRef>, EdgeKeyHash> M;
    buildEdgeMap(T,M);

    vector<BoundaryEdgeInfo> out; out.reserve(M.size());
    for (const auto& kv: M){
        int u=kv.first.u, v=kv.first.v;
        if (u<0||v<0||u>=(int)all.size()||v>=(int)all.size()) continue;
        if (labels[u]<0 || labels[v]<0) continue;
        if (labels[u]==labels[v]) continue;
        bool hull = (kv.second.size()==1);
        double len = geom::norm(all[v]-all[u]);
        Vec2 mid = (all[u]+all[v])*0.5;
        out.push_back({u,v,len,hull,mid});
    }
    return out;
}

inline vector<Vec2> orderByMST(const vector<Vec2>&pts){
    auto &C = cfg::get();
    int n=(int)pts.size(); if(n<=2) return pts;
    int K=std::min(C.knn_k, n-1);

    vector<vector<pair<int,double>>> adj(n);
    for(int i=0;i<n;i++){
        vector<pair<double,int>> cand; cand.reserve(n-1);
        for(int j=0;j<n;j++) if(i!=j){
            double d2=(pts[i].x-pts[j].x)*(pts[i].x-pts[j].x) + (pts[i].y-pts[j].y)*(pts[i].y-pts[j].y);
            cand.push_back({d2,j});
        }
        if((int)cand.size()>K){
            std::nth_element(cand.begin(), cand.begin()+K, cand.end(),
                             [](auto&A,auto&B){return A.first<B.first;});
            cand.resize(K);
        }
        for(auto &c: cand){
            double w=std::sqrt(std::max(0.0,c.first));
            adj[i].push_back({c.second,w});
            adj[c.second].push_back({i,w});
        }
    }
    vector<double> key(n,1e300); vector<int> par(n,-1); vector<char> in(n,0); key[0]=0;
    for(int it=0; it<n; ++it){
        int u=-1; double best=1e301;
        for(int i=0;i<n;i++) if(!in[i] && key[i]<best){ best=key[i]; u=i; }
        if(u==-1) break; in[u]=1;
        for(auto [v,w]: adj[u]) if(!in[v] && w<key[v]){ key[v]=w; par[v]=u; }
    }
    vector<vector<int>> tree(n);
    for(int v=0; v<n; v++) if(par[v]>=0){ tree[v].push_back(par[v]); tree[par[v]].push_back(v); }

    auto bfs = [&](int s){
        vector<double> d(n,1e300); vector<int> p(n,-1); std::queue<int> q; q.push(s); d[s]=0;
        while(!q.empty()){
            int u=q.front(); q.pop();
            for(int v: tree[u]) if(d[v]>1e299){
                double w=std::hypot(pts[u].x-pts[v].x, pts[u].y-pts[v].y);
                d[v]=d[u]+w; p[v]=u; q.push(v);
            }
        }
        int far=s; for(int i=0;i<n;i++) if(d[i]>d[far]) far=i;
        return std::tuple<int, vector<int>, vector<double>>(far,p,d);
    };
    auto [s1,p1,d1]=bfs(0);
    auto [s2,p2,d2]=bfs(s1);

    vector<int> path; for(int v=s2; v!=-1; v=p2[v]) path.push_back(v);
    vector<char> used(n,0); vector<Vec2> out; out.reserve(n);
    for(int id: path){ out.push_back(pts[id]); used[id]=1; }
    for(int i=0;i<n;i++) if(!used[i]) out.push_back(pts[i]);
    return out;
}

inline vector<int> orderIndicesByMST(const vector<Vec2>&pts){
    // 동일 알고리즘을 인덱스 버전으로
    auto ordered = orderByMST(pts);
    // 매칭: O(n^2)지만 n이 크지 않다는 가정
    vector<int> idx; idx.reserve(pts.size());
    vector<char> used(pts.size(),0);
    for (auto &p: ordered){
        int pick=-1; double best=1e300;
        for (int i=0;i<(int)pts.size();++i){ if(used[i]) continue;
            double dx=pts[i].x-p.x, dy=pts[i].y-p.y; double d2=dx*dx+dy*dy;
            if (d2<best){best=d2; pick=i;}
        }
        if(pick<0) pick=0;
        used[pick]=1; idx.push_back(pick);
    }
    return idx;
}

// --------- 1D Natural Cubic Spline ----------
struct Spline1D{
    vector<double> s,a,b,c,d;
    static void triSolve(vector<double>&dl, vector<double>&dm, vector<double>&du, vector<double>&rhs){
        int n=(int)dm.size();
        for(int i=1;i<n;++i){ double w=dl[i-1]/dm[i-1]; dm[i]-=w*du[i-1]; rhs[i]-=w*rhs[i-1]; }
        rhs[n-1]/=dm[n-1]; for(int i=n-2;i>=0;--i) rhs[i]=(rhs[i]-du[i]*rhs[i+1])/dm[i];
    }
    void fit(const vector<double>&_s, const vector<double>&y){
        int n=(int)_s.size(); s=_s; a=y; b.assign(n,0.0); c.assign(n,0.0); d.assign(n,0.0);
        if(n<3){ if(n==2) b[0]=(a[1]-a[0])/std::max(1e-30, s[1]-s[0]); return; }
        vector<double> h(n-1); for(int i=0;i<n-1;++i) h[i]=std::max(1e-30, s[i+1]-s[i]);
        vector<double> dl(n-2),dm(n-2),du(n-2),rhs(n-2);
        for(int i=1;i<=n-2;++i){ double hi_1=h[i-1], hi=h[i];
            dl[i-1]=hi_1; dm[i-1]=2.0*(hi_1+hi); du[i-1]=hi;
            rhs[i-1]=3.0*((a[i+1]-a[i])/hi - (a[i]-a[i-1])/hi_1);
        }
        if(n-2>0) triSolve(dl,dm,du,rhs);
        for(int i=1;i<=n-2;++i) c[i]=rhs[i-1]; c[0]=0.0; c[n-1]=0.0;
        for(int i=0;i<n-1;++i){
            b[i]=(a[i+1]-a[i])/h[i] - (2.0*c[i]+c[i+1])*h[i]/3.0;
            d[i]=(c[i+1]-c[i])/(3.0*h[i]);
        }
    }
    double eval(double si) const{
        int n=(int)s.size(); if(n==0) return 0.0; if(n==1) return a[0];
        int lo=0, hi=n-1;
        if(si<=s.front()) lo=0;
        else if(si>=s.back()) lo=n-2;
        else{ while(hi-lo>1){ int mid=(lo+hi)>>1; if(s[mid]<=si) lo=mid; else hi=mid; } }
        double t=si - s[lo]; return a[lo] + b[lo]*t + c[lo]*t*t + d[lo]*t*t*t;
    }
    void eval_with_deriv(double si, double&f, double&fp, double&fpp) const{
        int n=(int)s.size(); if(n==0){ f=fp=fpp=0; return; } if(n==1){ f=a[0]; fp=fpp=0; return; }
        int lo=0, hi=n-1;
        if(si<=s.front()) lo=0;
        else if(si>=s.back()) lo=n-2;
        else{ while(hi-lo>1){ int mid=(lo+hi)>>1; if(s[mid]<=si) lo=mid; else hi=mid; } }
        double t=si - s[lo];
        f = a[lo] + b[lo]*t + c[lo]*t*t + d[lo]*t*t*t;
        fp= b[lo] + 2.0*c[lo]*t + 3.0*d[lo]*t*t;
        fpp=2.0*c[lo] + 6.0*d[lo]*t;
    }
};

inline vector<Vec2> splineUniformResample(
    const vector<Vec2>& ordered, int samples, int paddingK, bool close_loop,
    Spline1D& spx_out, Spline1D& spy_out, double& s0_out, double& L_out)
{
    int N=(int)ordered.size(); if(N<3) return ordered;
    vector<Vec2> P; P.reserve(N+2*paddingK);
    for(int i=0;i<paddingK;++i) P.push_back(ordered[N-paddingK+i]);
    for(auto&q: ordered) P.push_back(q);
    for(int i=0;i<paddingK;++i) P.push_back(ordered[i]);

    int M=(int)P.size();
    vector<double> s(M,0.0), xs(M), ys(M);
    for(int i=1;i<M;++i){ double dx=P[i].x-P[i-1].x, dy=P[i].y-P[i-1].y; s[i]=s[i-1]+std::sqrt(dx*dx+dy*dy); }
    for(int i=0;i<M;++i){ xs[i]=P[i].x; ys[i]=P[i].y; }

    Spline1D spx,spy; spx.fit(s,xs); spy.fit(s,ys);
    double s0=s[paddingK], s1=s[M-paddingK-1], L=std::max(1e-30, s1-s0);

    vector<Vec2> out; out.reserve(samples + (close_loop?1:0));
    for(int k=0;k<samples;k++){ double si=s0 + L*(double(k)/double(samples));
        out.push_back({spx.eval(si), spy.eval(si)});
    }
    if(close_loop) out.push_back(out.front());

    spx_out=std::move(spx); spy_out=std::move(spy); s0_out=s0; L_out=L;
    return out;
}
} // namespace centerline

// =========================== Ray–Segment Utils ============================
static bool rayIntersectSegment(const geom::Vec2&A, const geom::Vec2&d,
                                const geom::Vec2&S0, const geom::Vec2&S1,
                                double &t_out, double eps=1e-15)
{
    double vx=S1.x-S0.x, vy=S1.y-S0.y;
    double den = d.x*(-vy) + d.y*(vx);
    if (std::fabs(den) < eps) return false;
    double ax=S0.x-A.x, ay=S0.y-A.y; double inv=1.0/den;
    double t = (ax*(-vy) + ay*(vx)) * inv;
    double u = (d.x*ay - d.y*ax) * inv;
    if (t>=0.0 && u>=-1e-12 && u<=1.0+1e-12){ t_out=t; return true; }
    return false;
}
static double rayToRingDistance(const geom::Vec2&P, const geom::Vec2&dir,
                                const vector<pair<geom::Vec2,geom::Vec2>>& ringEdges)
{
    double best=std::numeric_limits<double>::infinity();
    for (auto &e: ringEdges){
        double t; if (rayIntersectSegment(P,dir,e.first,e.second,t))
            if (t>0.0 && t<best) best=t;
    }
    return best;
}
static double minDistanceToSegments_global(const geom::Vec2&P, const vector<pair<geom::Vec2,geom::Vec2>>&E){
    double best=std::numeric_limits<double>::infinity();
    for (auto&e:E){
        geom::Vec2 a=e.first, b=e.second;
        geom::Vec2 ab{b.x-a.x,b.y-a.y}, ap{P.x-a.x,P.y-a.y};
        double denom=std::max(1e-30, ab.x*ab.x+ab.y*ab.y);
        double t=std::clamp((ab.x*ap.x+ab.y*ap.y)/denom, 0.0, 1.0);
        geom::Vec2 Q{a.x+ab.x*t, a.y+ab.y*t};
        best=std::min(best, std::hypot(P.x-Q.x,P.y-Q.y));
    }
    return best;
}
static void distancesToRings(const geom::Vec2&P, const geom::Vec2&n,
                             const vector<pair<geom::Vec2,geom::Vec2>>& innerE,
                             const vector<pair<geom::Vec2,geom::Vec2>>& outerE,
                             double&d_inner, double&d_outer)
{
    geom::Vec2 npos=n, nneg{-n.x,-n.y};
    double di1=rayToRingDistance(P,npos,innerE), di2=rayToRingDistance(P,nneg,innerE);
    d_inner = (std::isfinite(di1)||std::isfinite(di2))? std::min(di1,di2): minDistanceToSegments_global(P,innerE);
    double do1=rayToRingDistance(P,npos,outerE), do2=rayToRingDistance(P,nneg,outerE);
    d_outer = (std::isfinite(do1)||std::isfinite(do2))? std::min(do1,do2): minDistanceToSegments_global(P,outerE);
    if(!std::isfinite(d_inner)) d_inner=0.0; if(!std::isfinite(d_outer)) d_outer=0.0;
}
static inline geom::Vec2 dir_from_heading_rad(double rad){ return {std::cos(rad), std::sin(rad)}; }

// =============================== Timing ===================================
namespace timing {
using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;
struct ScopedAcc{
    const char* name; double* acc; Clock::time_point t0;
    ScopedAcc(const char*n,double*a):name(n),acc(a),t0(Clock::now()){}
    ~ScopedAcc(){ auto ms=std::chrono::duration_cast<Ms>(Clock::now()-t0).count();
        if(acc) *acc += ms; std::cerr.setf(std::ios::fixed);
        std::cerr<<"[TIME] "<<name<<" = "<<std::setprecision(3)<<ms<<" ms\n";
    }
};
} // namespace timing

// ============================== Raceline (min-curv) =======================
namespace raceline_min_curv {
using geom::Vec2; using timing::ScopedAcc;

struct DiffOps{
    int N; double h, inv2h, invh2;
    DiffOps(int N_, double h_):N(N_),h(h_),inv2h(1.0/(2*h)),invh2(1.0/(h*h)){}
    inline int wrap(int i)const{ i%=N; if(i<0) i+=N; return i; }
    void D1(const vector<double>&a, vector<double>&out)const{
        out.resize(N); for(int i=0;i<N;++i){ int ip=wrap(i+1), im=wrap(i-1); out[i]=(a[ip]-a[im])*inv2h; }
    }
    void D2(const vector<double>&a, vector<double>&out)const{
        out.resize(N); for(int i=0;i<N;++i){ int ip=wrap(i+1), im=wrap(i-1); out[i]=(a[ip]-2*a[i]+a[im])*invh2; }
    }
    void D1T(const vector<double>&v, vector<double>&out)const{
        out.resize(N); for(int i=0;i<N;++i){ int im=wrap(i-1), ip=wrap(i+1); out[i]=(v[im]-v[ip])*inv2h; }
    }
    void D2T(const vector<double>&v, vector<double>&out)const{ D2(v,out); }
};
struct DiffOpsOpen{
    int N; double h, invh, inv2h, invh2;
    DiffOpsOpen(int N_, double h_):N(N_),h(h_),invh(1.0/h),inv2h(1.0/(2*h)),invh2(1.0/(h*h)){}
    void D1(const vector<double>&a, vector<double>&out)const{
        out.assign(N,0.0); if(N==0) return; if(N==1){out[0]=0;return;}
        out[0]=(a[1]-a[0])*invh; for(int i=1;i<=N-2;++i) out[i]=(a[i+1]-a[i-1])*inv2h; out[N-1]=(a[N-1]-a[N-2])*invh;
    }
    void D1T(const vector<double>&v, vector<double>&out)const{
        out.assign(N,0.0); if(N<=1) return;
        out[0] += (-invh)*v[0]; out[1] += (+invh)*v[0];
        for(int i=1;i<=N-2;++i){ out[i-1]+=(-inv2h)*v[i]; out[i+1]+=(+inv2h)*v[i]; }
        out[N-2]+=(-invh)*v[N-1]; out[N-1]+=(+invh)*v[N-1];
    }
    void D2(const vector<double>&a, vector<double>&out)const{
        out.assign(N,0.0); if(N<=2) return; for(int i=1;i<=N-2;++i) out[i]=(a[i+1]-2*a[i]+a[i-1])*invh2;
    }
    void D2T(const vector<double>&v, vector<double>&out)const{
        out.assign(N,0.0); if(N<=2) return; for(int i=1;i<=N-2;++i){ out[i-1]+= (+invh2)*v[i]; out[i]+= (-2*invh2)*v[i]; out[i+1]+= (+invh2)*v[i]; }
    }
};

static void normals_from_points_generic(const vector<Vec2>&P, bool closed, vector<Vec2>&n){
    int N=(int)P.size(); n.assign(N,{0,0}); if(N==0) return;
    auto t_of = [&](int i)->Vec2{
        if(N==1) return {1,0};
        if(closed){ int ip=(i+1)%N, im=(i-1+N)%N; return {(P[ip].x-P[im].x)*0.5, (P[ip].y-P[im].y)*0.5}; }
        else{
            if(i==0) return {P[1].x-P[0].x, P[1].y-P[0].y};
            if(i==N-1) return {P[N-1].x-P[N-2].x, P[N-1].y-P[N-2].y};
            return {(P[i+1].x-P[i-1].x)*0.5, (P[i+1].y-P[i-1].y)*0.5};
        }
    };
    for(int i=0;i<N;++i){ Vec2 t=t_of(i); if(geom::norm(t)<1e-15) t={1,0}; Vec2 nv{-t.y,t.x}; n[i]=geom::normalize(nv,1e-15); }
}

static void heading_curv_from_points_generic(const vector<Vec2>&P, double h, bool closed,
                                             vector<double>&heading, vector<double>&kappa)
{
    int N=(int)P.size(); heading.assign(N,0.0); kappa.assign(N,0.0); if(N==0) return;
    auto deriv = [&](int i,double&xp,double&yp,double&xpp,double&ypp){
        if(N==1){ xp=1; yp=0; xpp=ypp=0; return; }
        if(closed){ int ip=(i+1)%N, im=(i-1+N)%N;
            xp=(P[ip].x-P[im].x)/(2*h); yp=(P[ip].y-P[im].y)/(2*h);
            xpp=(P[ip].x-2*P[i].x+P[im].x)/(h*h); ypp=(P[ip].y-2*P[i].y+P[im].y)/(h*h);
        } else {
            if(i==0){ xp=(P[1].x-P[0].x)/h; yp=(P[1].y-P[0].y)/h;
                if(N>=3){ xpp=(P[2].x-2*P[1].x+P[0].x)/(h*h); ypp=(P[2].y-2*P[1].y+P[0].y)/(h*h);} else xpp=ypp=0;
            } else if(i==N-1){ xp=(P[N-1].x-P[N-2].x)/h; yp=(P[N-1].y-P[N-2].y)/h;
                if(N>=3){ xpp=(P[N-1].x-2*P[N-2].x+P[N-3].x)/(h*h); ypp=(P[N-1].y-2*P[N-2].y+P[N-3].y)/(h*h);} else xpp=ypp=0;
            } else { xp=(P[i+1].x-P[i-1].x)/(2*h); yp=(P[i+1].y-P[i-1].y)/(2*h);
                xpp=(P[i+1].x-2*P[i].x+P[i-1].x)/(h*h); ypp=(P[i+1].y-2*P[i].y+P[i-1].y)/(h*h);
            }
        }
    };
    for(int i=0;i<N;++i){
        double xp,yp,xpp,ypp; deriv(i,xp,yp,xpp,ypp);
        heading[i]=std::atan2(yp,xp);
        double denom=std::pow(std::max(1e-12, xp*xp+yp*yp), 1.5);
        kappa[i]=(xp*ypp - yp*xpp)/denom;
    }
}

struct LinGeom{ vector<double> A1,A2,N0,W; };
static LinGeom precompute_lin_geom_generic(const vector<Vec2>&P, const vector<Vec2>&n, double h, bool closed){
    int N=(int)P.size(); vector<double> xp(N),yp(N),xpp(N),ypp(N);
    auto deriv = [&](int i,double&_xp,double&_yp,double&_xpp,double&_ypp){
        if(N==1){ _xp=1; _yp=0; _xpp=_ypp=0; return; }
        if(closed){ int ip=(i+1)%N, im=(i-1+N)%N;
            _xp=(P[ip].x-P[im].x)/(2*h); _yp=(P[ip].y-P[im].y)/(2*h);
            _xpp=(P[ip].x-2*P[i].x+P[im].x)/(h*h); _ypp=(P[ip].y-2*P[i].y+P[im].y)/(h*h);
        } else {
            if(i==0){ _xp=(P[1].x-P[0].x)/h; _yp=(P[1].y-P[0].y)/h;
                if(N>=3){ _xpp=(P[2].x-2*P[1].x+P[0].x)/(h*h); _ypp=(P[2].y-2*P[1].y+P[0].y)/(h*h);} else _xpp=_ypp=0;
            } else if(i==N-1){ _xp=(P[N-1].x-P[N-2].x)/h; _yp=(P[N-1].y-P[N-2].y)/h;
                if(N>=3){ _xpp=(P[N-1].x-2*P[N-2].x+P[N-3].x)/(h*h); _ypp=(P[N-1].y-2*P[N-2].y+P[N-3].y)/(h*h);} else _xpp=_ypp=0;
            } else { _xp=(P[i+1].x-P[i-1].x)/(2*h); _yp=(P[i+1].y-P[i-1].y)/(2*h);
                _xpp=(P[i+1].x-2*P[i].x+P[i-1].x)/(h*h); _ypp=(P[i+1].y-2*P[i].y+P[i-1].y)/(h*h);
            }
        }
    };
    for(int i=0;i<N;++i) deriv(i,xp[i],yp[i],xpp[i],ypp[i]);

    LinGeom G; G.A1.resize(N); G.A2.resize(N); G.N0.resize(N); G.W.resize(N);
    for(int i=0;i<N;++i){
        G.A1[i]= n[i].x*ypp[i] - n[i].y*xpp[i];
        G.A2[i]= xp[i]*n[i].y - yp[i]*n[i].x;
        G.N0[i]= xp[i]*ypp[i] - yp[i]*xpp[i];
        double denom=std::pow(std::max(1e-12, xp[i]*xp[i]+yp[i]*yp[i]), 1.5);
        G.W[i]=1.0/denom;
    }
    return G;
}

struct CostGrad{ double J; vector<double> grad; };
static CostGrad eval_cost_grad_frozen(const vector<double>&A1,const vector<double>&A2,
                                      const vector<double>&N0,const vector<double>&W,
                                      double h,double lambda_smooth,const vector<double>&alpha,bool closed)
{
    int N=(int)alpha.size(); vector<double> a1,a2;
    if(closed){ DiffOps D(N,h); D.D1(alpha,a1); D.D2(alpha,a2); }
    else{ DiffOpsOpen D(N,h); D.D1(alpha,a1); D.D2(alpha,a2); }

    vector<double> z(N); for(int i=0;i<N;++i) z[i] = W[i]*(N0[i] + A1[i]*a1[i] + A2[i]*a2[i]);

    double J=0; for(double v: z) J+=v*v;
    double Jsm=0; for(double v: a1) Jsm+=v*v;
    J += lambda_smooth * Jsm;

    vector<double> Wz(N),q1(N),q2(N), g1,g2,gsm, D1a;
    for(int i=0;i<N;++i){ Wz[i]=W[i]*z[i]; q1[i]=A1[i]*Wz[i]; q2[i]=A2[i]*Wz[i]; }
    if(closed){ DiffOps D(N,h); D.D1T(q1,g1); D.D2T(q2,g2); D.D1(alpha,D1a); D.D1T(D1a,gsm); }
    else{ DiffOpsOpen D(N,h); D.D1T(q1,g1); D.D2T(q2,g2); D.D1(alpha,D1a); D.D1T(D1a,gsm); }

    vector<double> grad(N); for(int i=0;i<N;++i) grad[i]=2.0*(g1[i]+g2[i]) + 2.0*lambda_smooth*gsm[i];
    return {J,std::move(grad)};
}

struct Result{
    vector<Vec2> raceline;
    vector<double> heading, curvature;
    vector<double> alpha_total, alpha_last;
};

static Result compute_min_curvature_raceline(const vector<Vec2>&center,
                                             const vector<pair<Vec2,Vec2>>& innerE,
                                             const vector<pair<Vec2,Vec2>>& outerE,
                                             double veh_width, double L, bool closed)
{
    auto &C = cfg::get();
    int N=(int)center.size(); if(N==0) return {};
    double h = L / double(N);

    vector<Vec2> P = center, n; normals_from_points_generic(P, closed, n);

    auto safe_ray = [&](const Vec2&P0, const Vec2&dir, const vector<pair<Vec2,Vec2>>&E){
        double t = rayToRingDistance(P0,dir,E);
        if(!std::isfinite(t)) t = minDistanceToSegments_global(P0,E);
        if(!std::isfinite(t)) t = 0.0;
        return std::max(0.0, t);
    };

    vector<double> lo(N,0.0), hi(N,0.0);
    for(int i=0;i<N;++i){
        Vec2 nv=n[i], P0=P[i], nneg{-nv.x,-nv.y};
        double dpos = std::min(safe_ray(P0,nv,innerE), safe_ray(P0,nv,outerE));
        double dneg = std::min(safe_ray(P0,nneg,innerE), safe_ray(P0,nneg,outerE));
        double guard = veh_width*0.5 + C.safety_margin_m;
        hi[i]=std::max(0.0, dpos-guard);
        lo[i]=-std::max(0.0, dneg-guard);
        if(!std::isfinite(hi[i])) hi[i]=0.0;
        if(!std::isfinite(lo[i])) lo[i]=0.0;
    }

    if (C.verbose){
        double hi_avg=0, lo_avg=0, hi_max=0, lo_max=0;
        for(int i=0;i<N;++i){ hi_avg+=hi[i]; lo_avg+=-lo[i]; hi_max=std::max(hi_max,hi[i]); lo_max=std::max(lo_max,-lo[i]); }
        hi_avg/=std::max(1,N); lo_avg/=std::max(1,N);
        cerr<<"[corridor] mean+ "<<hi_avg<<" mean- "<<lo_avg<<" max+ "<<hi_max<<" max- "<<lo_max<<"\n";
    }

    vector<double> alpha(N,0.0), alpha_accum(N,0.0), alpha_last(N,0.0);
    for(int outer=0; outer<C.max_outer_iters; ++outer){
        auto G = precompute_lin_geom_generic(P, n, h, closed);
        double step=C.step_init;
        auto cg = eval_cost_grad_frozen(G.A1,G.A2,G.N0,G.W,h,C.lambda_smooth,alpha,closed);
        double J_prev=cg.J; if(C.verbose) cerr<<"[GN "<<outer<<"] J0="<<J_prev<<" step="<<step<<" lambda="<<C.lambda_smooth<<"\n";

        for(int it=0; it<C.max_inner_iters; ++it){
            bool accepted=false; int bt=0;
            while(bt<20){
                vector<double> a_new(N);
                for(int i=0;i<N;++i){ double ai=alpha[i] - step*cg.grad[i]; a_new[i]=std::min(hi[i], std::max(lo[i], ai)); }
                auto cg_new = eval_cost_grad_frozen(G.A1,G.A2,G.N0,G.W,h,C.lambda_smooth,a_new,closed);
                double dec=0.0; for(int i=0;i<N;++i) dec += cg.grad[i]*(a_new[i]-alpha[i]);
                if (cg_new.J <= cg.J + C.armijo_c*dec){
                    alpha.swap(a_new); cg=std::move(cg_new); accepted=true; break;
                }
                step*=0.5; bt++; if(step<C.step_min) break;
            }
            if(!accepted) break;
            if(std::fabs(J_prev - cg.J) < 1e-10) break;
            J_prev=cg.J;
        }
        alpha_last=alpha;

        for(int i=0;i<N;++i){ P[i].x += n[i].x*alpha[i]; P[i].y += n[i].y*alpha[i]; alpha_accum[i]+=alpha[i]; }
        normals_from_points_generic(P,closed,n);

        // corridor update
        for(int i=0;i<N;++i){
            Vec2 nv=n[i], P0=P[i], nneg{-nv.x,-nv.y};
            double dpos = std::min(safe_ray(P0,nv,innerE), safe_ray(P0,nv,outerE));
            double dneg = std::min(safe_ray(P0,nneg,innerE), safe_ray(P0,nneg,outerE));
            double guard = cfg::get().veh_width_m*0.5 + cfg::get().safety_margin_m;
            hi[i]=std::max(0.0, dpos-guard); lo[i]=-std::max(0.0, dneg-guard);
            if(!std::isfinite(hi[i])) hi[i]=0.0; if(!std::isfinite(lo[i])) lo[i]=0.0;
        }
        std::fill(alpha.begin(), alpha.end(), 0.0);
    }

    vector<double> heading, kappa;
    heading_curv_from_points_generic(P, h, closed, heading, kappa);

    return {std::move(P), std::move(heading), std::move(kappa), std::move(alpha_accum), std::move(alpha_last)};
}
} // namespace raceline_min_curv

// ============================ Raceline (min-time) =========================
namespace raceline_min_time {
using geom::Vec2;
using raceline_min_curv::DiffOps;
using raceline_min_curv::DiffOpsOpen;
using raceline_min_curv::precompute_lin_geom_generic;
using timing::ScopedAcc;

struct VelProfile {
    vector<double> v, ax;
    double lap_time = 0.0;
};

static inline double clamp01(double x){ return x<0?0:(x>1?1:x); }

static VelProfile velocity_profile_forward_backward(
    const vector<double>& kappa, double h, bool closed)
{
    auto &C = cfg::get();
    int N=(int)kappa.size(); if(N==0) return {};
    vector<double> v(N, C.v_cap_mps);

    // 곡률 한계 속도
    for(int i=0;i<N;++i){
        double k = std::fabs(kappa[i]);
        double v_kappa = std::sqrt(C.a_lat_max / std::max(k, C.kappa_eps));
        v[i] = std::min(v[i], v_kappa);
    }

    // 반복: 전/후진 패스 (마찰원 기반 종/제동 한계)
    auto ax_max_at = [&](double vi, double ki)->std::pair<double,double>{
        // 사용 횡가속
        double alat = vi*vi*std::fabs(ki);
    
        // 총가속 한계(일관성 보장)
        double a_total = cfg::get().use_total_ge_lat
                       ? std::max(cfg::get().a_total_max, cfg::get().a_lat_max)
                       : cfg::get().a_total_max;
    
        // 남는 종가속 여유(마찰원)
        double a_res = std::sqrt(std::max(0.0, a_total*a_total - alat*alat));
    
        // 항력 + 구름저항
        double Fd = 0.5*cfg::get().rho_air*cfg::get().Cd*cfg::get().A_front_m2*vi*vi;
        double Fr = cfg::get().mass_kg*9.81*cfg::get().c_rr;
    
        // 파워 리미트
        double a_power = (cfg::get().P_max_W>0 && vi>1e-6)
                       ? (cfg::get().P_max_W/(cfg::get().mass_kg*vi) - (Fd+Fr)/cfg::get().mass_kg)
                       : 1e9;
    
        double a_acc = std::min({a_res, cfg::get().a_long_acc_cap, a_power});
        a_acc = std::max(0.0, a_acc);
    
        double a_brk = std::min(a_res, cfg::get().a_long_brake_cap) + (Fd+Fr)/cfg::get().mass_kg;
        a_brk = std::max(0.0, a_brk);
        return {a_acc, a_brk};
    };    

    int iters=C.max_vpass_iters;
    while(iters--){
        // forward(가속)
        for(int i=0;i+1<N;++i){
            auto [a_acc, a_brk] = ax_max_at(v[i], kappa[i]);
            double vf = std::sqrt(std::max(0.0, v[i]*v[i] + 2.0*a_acc*h));
            v[i+1] = std::min(v[i+1], vf);
        }
        if (closed){
            // 루프 연속성 보정 한 사이클
            auto [a_acc, a_brk] = ax_max_at(v[N-1], kappa[N-1]);
            double vf0 = std::sqrt(std::max(0.0, v[N-1]*v[N-1] + 2.0*a_acc*h));
            v[0] = std::min(v[0], vf0);
        }
        // backward(제동)
        for(int i=N-2;i>=0;--i){
            auto [a_acc, a_brk] = ax_max_at(v[i+1], kappa[i+1]);
            double vb = std::sqrt(std::max(0.0, v[i+1]*v[i+1] + 2.0*a_brk*h));
            v[i] = std::min(v[i], vb);
        }
        if (closed){
            auto [a_acc, a_brk] = ax_max_at(v[0], kappa[0]);
            double vbN = std::sqrt(std::max(0.0, v[0]*v[0] + 2.0*a_brk*h));
            v[N-1] = std::min(v[N-1], vbN);
        }
    }

    // ax, lap time
    vector<double> ax(N,0.0); double t=0.0;
    for(int i=0;i<N;++i){
        int j = (i+1<N)? i+1 : (closed?0:i);
        double v0=v[i], v1=v[j];
        ax[i] = (v1*v1 - v0*v0) / (2.0*h);
        t += h / std::max(1e-6, v[i]);
    }
    return {std::move(v), std::move(ax), t};
}

// time-weighted 최소곡률 스텝의 코스트/그라디언트
struct CostGrad{ double J; vector<double> grad; };
static CostGrad eval_cost_grad_timeweighted(
    const vector<double>&A1,const vector<double>&A2,
    const vector<double>&N0,const vector<double>&W,
    const vector<double>&gamma2, double h, double lambda_smooth,
    const vector<double>&alpha, bool closed)
{
    int N=(int)alpha.size();
    vector<double> a1,a2;
    if(closed){ DiffOps D(N,h); D.D1(alpha,a1); D.D2(alpha,a2); }
    else{ DiffOpsOpen D(N,h); D.D1(alpha,a1); D.D2(alpha,a2); }

    // r = W * (N0 + A1 a1 + A2 a2)
    vector<double> r(N), z(N);
    for(int i=0;i<N;++i) r[i] = W[i]*(N0[i] + A1[i]*a1[i] + A2[i]*a2[i]);

    // J = sum gamma^2 * r^2 + lambda * ||a1||^2
    double J=0; for(int i=0;i<N;++i) J += gamma2[i]*r[i]*r[i];
    double Jsm=0; for(double v: a1) Jsm += v*v; J += lambda_smooth * Jsm;

    // grad = 2 * (D1^T (A1 .* (W .* (gamma^2 .* r))) + D2^T (A2 .* (W .* (gamma^2 .* r)))) + 2λ D1^T D1 α
    vector<double> Wg_r(N), q1(N), q2(N), g1,g2, D1a, gsm;
    for(int i=0;i<N;++i){ Wg_r[i] = W[i] * gamma2[i] * r[i]; q1[i]=A1[i]*Wg_r[i]; q2[i]=A2[i]*Wg_r[i]; }

    if(closed){ DiffOps D(N,h); D.D1T(q1,g1); D.D2T(q2,g2); D.D1(alpha,D1a); D.D1T(D1a,gsm); }
    else{ DiffOpsOpen D(N,h); D.D1T(q1,g1); D.D2T(q2,g2); D.D1(alpha,D1a); D.D1T(D1a,gsm); }

    vector<double> grad(N);
    for(int i=0;i<N;++i) grad[i] = 2.0*(g1[i]+g2[i]) + 2.0*lambda_smooth*gsm[i];
    return {J, std::move(grad)};
}

struct Result{
    vector<Vec2> raceline;
    vector<double> heading, curvature;
    vector<double> alpha_total, alpha_last;
    vector<double> v, ax;
    double lap_time = 0.0;
};

static Result compute_min_time_raceline(
    const vector<Vec2>& center,
    const vector<pair<Vec2,Vec2>>& innerE,
    const vector<pair<Vec2,Vec2>>& outerE,
    double veh_width, double L, bool closed)
{
    auto &C = cfg::get();
    int N=(int)center.size(); if(N==0) return {};
    double h = L / double(N);

    vector<Vec2> P = center, n; raceline_min_curv::normals_from_points_generic(P, closed, n);

    auto safe_ray = [&](const Vec2&P0, const Vec2&dir, const vector<pair<Vec2,Vec2>>&E){
        double t = rayToRingDistance(P0,dir,E);
        if(!std::isfinite(t)) t = minDistanceToSegments_global(P0,E);
        if(!std::isfinite(t)) t = 0.0;
        return std::max(0.0, t);
    };

    // 초기 코리도
    vector<double> lo(N,0.0), hi(N,0.0);
    for(int i=0;i<N;++i){
        Vec2 nv=n[i], P0=P[i], nneg{-nv.x,-nv.y};
        double dpos = std::min(safe_ray(P0,nv,innerE), safe_ray(P0,nv,outerE));
        double dneg = std::min(safe_ray(P0,nneg,innerE), safe_ray(P0,nneg,outerE));
        double guard = veh_width*0.5 + C.safety_margin_m;
        hi[i]=std::max(0.0, dpos-guard);
        lo[i]=-std::max(0.0, dneg-guard);
        if(!std::isfinite(hi[i])) hi[i]=0.0;
        if(!std::isfinite(lo[i])) lo[i]=0.0;
    }

    vector<double> alpha(N,0.0), alpha_accum(N,0.0), alpha_last(N,0.0);

    for(int outer=0; outer<C.max_outer_iters; ++outer){
        // 선형화(곡률) 준비
        auto G = precompute_lin_geom_generic(P, n, h, closed);
        // 현재 곡률(실좌표) 계산
        vector<double> heading, kappa;
        raceline_min_curv::heading_curv_from_points_generic(P, h, closed, heading, kappa);

        // v(s) 프로파일 (전/후진 패스)
        auto VP = velocity_profile_forward_backward(kappa, h, closed);

        // time-weight 생성
        vector<double> gamma2(N,1.0);
        double v_avg = 0.0; for (double vi : VP.v) v_avg += vi; v_avg /= std::max(1, N);

        double mean_r = 0.0, mean_gamma = 0.0; // 디버그용
        for (int i = 0; i < N; ++i) {
            double k = std::fabs(kappa[i]);
            double vkappa = std::sqrt(C.a_lat_max / std::max(k, C.kappa_eps));
            // 횡가속 사용률 r = a_lat/a_lat_max = (v^2 * |kappa|) / a_lat_max
            double r = std::pow( std::min(1.0, VP.v[i] / std::max(1e-6, vkappa)), 2.0 );
            r = std::min(1.0, std::max(0.0, r));  // clamp

            // (1) corner weight: (1 + w * r^p)
            double corner_w = 1.0 + C.w_time_gain * std::pow(r, C.time_gamma_power);

            // (2) slow segment weight: 1 + inv_v_gain * (v_avg / v - 1)  (>= 1)
            double invv_w = 1.0;
            if (C.time_weight_use_inv_v) {
                double ratio = v_avg / std::max(1e-6, VP.v[i]);
                invv_w = 1.0 + C.inv_v_gain * (ratio - 1.0);
                if (invv_w < 1.0) invv_w = 1.0;   // 안정화
                if (invv_w > 3.0) invv_w = 3.0;   // 과도 방지
            }

            double gamma = corner_w * invv_w;
            gamma2[i] = gamma * gamma;

            mean_r += r; mean_gamma += gamma;
        }
        mean_r /= std::max(1, N);
        mean_gamma /= std::max(1, N);
        std::cerr << "[MT gamma] mean_r=" << std::fixed << std::setprecision(3) << mean_r
                << " mean_gamma=" << mean_gamma << "\n";

        // ===== ADD: 코너 포화율 디버그 =====
        int sat_cnt = 0;
        for (int i = 0; i < N; ++i) {
            double vkappa = std::sqrt(C.a_lat_max / std::max(std::fabs(kappa[i]), C.kappa_eps));
            if (VP.v[i] >= 0.99 * vkappa) ++sat_cnt;   // kappa 제한 속도에 99% 이상 근접
        }
        double sat_ratio = 100.0 * sat_cnt / std::max(1, N);
        std::cerr << "[MT sat] kappa-limited " << std::fixed << std::setprecision(1)
                << sat_ratio << "% of samples\n";
        // ===================================


        // 한 번의 time-weighted 최소곡률 스텝 (projected gradient + Armijo)
        double step=C.step_init;
        auto cg = eval_cost_grad_timeweighted(G.A1,G.A2,G.N0,G.W,gamma2,h,C.lambda_smooth,alpha,closed);
        double J_prev=cg.J; if(C.verbose) cerr<<"[MT "<<outer<<"] J0="<<J_prev<<" step="<<step<<" t="<<VP.lap_time<<"\n";

        for(int it=0; it<C.max_inner_iters; ++it){
            bool accepted=false; int bt=0;
            while(bt<20){
                vector<double> a_new(N);
                for(int i=0;i<N;++i){
                    double ai = alpha[i] - step*cg.grad[i];
                    a_new[i] = std::min(hi[i], std::max(lo[i], ai));
                }
                auto cg_new = eval_cost_grad_timeweighted(G.A1,G.A2,G.N0,G.W,gamma2,h,C.lambda_smooth,a_new,closed);
                double dec=0.0; for(int i=0;i<N;++i) dec += cg.grad[i]*(a_new[i]-alpha[i]);
                if (cg_new.J <= cg.J + C.armijo_c*dec){
                    alpha.swap(a_new); cg=std::move(cg_new); accepted=true; break;
                }
                step*=0.5; bt++; if(step<C.step_min) break;
            }
            if(!accepted) break;
             // ===== ADD: 수용된 스텝의 그라디언트 크기 디버그 =====
            double g2 = std::inner_product(cg.grad.begin(), cg.grad.end(),
                                           cg.grad.begin(), 0.0);
            std::cerr << "    [PG] J=" << cg.J
                        << "  step=" << step
                        << "  ||grad||^2=" << g2
                        << "  bt=" << bt << "\n";
            // ================================================
            if(std::fabs(J_prev - cg.J) < 1e-10) break;
            J_prev=cg.J;
        }
        alpha_last=alpha;

        // 경로 업데이트 + 코리도 업데이트
        for(int i=0;i<N;++i){ P[i].x += n[i].x*alpha[i]; P[i].y += n[i].y*alpha[i]; alpha_accum[i]+=alpha[i]; }
        raceline_min_curv::normals_from_points_generic(P, closed, n);

        for(int i=0;i<N;++i){
            Vec2 nv=n[i], P0=P[i], nneg{-nv.x,-nv.y};
            double dpos = std::min(safe_ray(P0,nv,innerE), safe_ray(P0,nv,outerE));
            double dneg = std::min(safe_ray(P0,nneg,innerE), safe_ray(P0,nneg,outerE));
            double guard = cfg::get().veh_width_m*0.5 + cfg::get().safety_margin_m;
            hi[i]=std::max(0.0, dpos-guard); lo[i]=-std::max(0.0, dneg-guard);
            if(!std::isfinite(hi[i])) hi[i]=0.0; if(!std::isfinite(lo[i])) lo[i]=0.0;
        }
        std::fill(alpha.begin(), alpha.end(), 0.0);
    }

    // 최종 기하/속도
    vector<double> heading, kappa;
    raceline_min_curv::heading_curv_from_points_generic(P, h, closed, heading, kappa);
    auto VP = velocity_profile_forward_backward(kappa, h, closed);

    return {std::move(P), std::move(heading), std::move(kappa),
            std::move(alpha_accum), std::move(alpha_last),
            std::move(VP.v), std::move(VP.ax), VP.lap_time};
}
} // namespace raceline_min_time


// =============================== Pipeline =================================
namespace pipeline {
using geom::Vec2;

struct Triangulation {
    vector<Vec2> all;
    vector<int>  label; // 0 inner, 1 outer
    vector<delaunay::Tri> tris;
};

static Triangulation buildDT(const vector<Vec2>& inner, const vector<Vec2>& outer){
    Triangulation R;
    R.all = inner; R.all.insert(R.all.end(), outer.begin(), outer.end());
    R.label.assign(R.all.size(), 0);
    for (size_t i=0;i<R.all.size();++i) R.label[i] = (i<inner.size()?0:1);
    R.tris = delaunay::bowyerWatson(R.all);
    return R;
}

struct MidsFiltered {
    vector<centerline::BoundaryEdgeInfo> binfo; // 전체(라벨-서로다름)
    vector<Vec2> mids;               // 길이필터 통과 중점
    vector<int>  keep_edge_idx;      // binfo 인덱스
};

static MidsFiltered extract_mids_with_len_filter(const Triangulation& T, const string& base){
    auto &C = cfg::get();
    MidsFiltered R;
    R.binfo = centerline::labelBoundaryEdges_with_len(T.all, T.tris, T.label);
    if (R.binfo.empty()) throw std::runtime_error("no label-different boundary edges");

    // 저장: 라벨-다른 엣지
    {
        std::unordered_map<delaunay::EdgeKey, vector<delaunay::EdgeRef>, delaunay::EdgeKeyHash> M;
        delaunay::buildEdgeMap(T.tris, M);
        vector<pair<int,int>> edges_all; edges_all.reserve(M.size());
        for (auto& kv: M) edges_all.push_back({kv.first.u, kv.first.v});
        io::saveCSV_edgesIdx(base + "_edges_all_idx.csv", edges_all);

        vector<pair<int,int>> edges_mixed; edges_mixed.reserve(R.binfo.size());
        for (auto &e: R.binfo) edges_mixed.push_back({e.u,e.v});
        io::saveCSV_edgesIdx(base + "_edges_labeldiff_idx.csv", edges_mixed);
        if (C.verbose) cerr << "[edges] label-different edges = " << edges_mixed.size() << "\n";
    }

    // 길이 중앙값 기반 컷오프
    vector<double> lens; lens.reserve(R.binfo.size());
    for (auto &e: R.binfo) lens.push_back(e.len);
    std::sort(lens.begin(), lens.end());
    auto quant = [&](double p)->double{
        if (lens.empty()) return 0.0;
        double idx = p * (lens.size()-1);
        size_t i=(size_t)std::floor(idx), j=std::min(i+1, lens.size()-1);
        double t=idx-i; return (1.0-t)*lens[i] + t*lens[j];
    };
    double Lmed = quant(0.5);
    double cutoff = std::min(C.boundary_edge_abs_max, C.boundary_edge_len_scale * std::max(1e-12, Lmed));

    // 필터
    R.mids.reserve(R.binfo.size()); R.keep_edge_idx.reserve(R.binfo.size());
    for (int i=0;i<(int)R.binfo.size();++i){
        const auto &e = R.binfo[i];
        if (!C.enable_boundary_len_filter || e.len <= cutoff){
            R.mids.push_back(e.mid);
            R.keep_edge_idx.push_back(i);
        }
    }

    if (R.mids.size()<2){
        io::saveCSV_pointsXY(base + "_mids_raw.csv", R.mids);
        throw std::runtime_error("not enough midpoints after length filter");
    }

    // 저장: kept edges
    {
        vector<pair<int,int>> edges_kept; edges_kept.reserve(R.keep_edge_idx.size());
        for (int idx: R.keep_edge_idx){ const auto &e=R.binfo[idx]; edges_kept.push_back({e.u,e.v}); }
        io::saveCSV_edgesIdx(base + "_edges_labeldiff_kept_idx.csv", edges_kept);
        if (C.verbose) cerr << "[edges] kept (len-filtered) = "<<edges_kept.size()<<" / "<<R.binfo.size()<<"\n";
    }

    return R;
}

struct OrderedMids {
    vector<Vec2> ordered;
    vector<int>  mids_order_idx; // mids 기준 인덱스
};

static OrderedMids order_and_align_mids_open_closed(const vector<Vec2>& mids, bool closed_mode){
    auto &C = cfg::get();
    OrderedMids R;
    R.ordered = centerline::orderByMST(mids);
    R.mids_order_idx = centerline::orderIndicesByMST(mids);

    auto reverse_both = [&](vector<Vec2>&A, vector<int>&B){ std::reverse(A.begin(),A.end()); std::reverse(B.begin(),B.end()); };
    auto dist2 = [](const Vec2&p,const Vec2&q){ double dx=p.x-q.x, dy=p.y-q.y; return dx*dx+dy*dy; };
    auto nearest_idx = [&](const vector<Vec2>&S, const Vec2&t)->size_t{
        size_t k=0; double best=1e300;
        for(size_t i=0;i<S.size();++i){ double d=dist2(S[i],t); if(d<best){best=d; k=i;} }
        return k;
    };
    auto vnorm = [](const Vec2&v)->Vec2{ return geom::normalize(v,1e-12); };
    auto local_dir_open = [&](const vector<Vec2>&S, size_t i)->Vec2{
        if(S.size()<2) return {1,0};
        if (i+1<S.size()) return vnorm(Vec2{S[i+1].x-S[i].x, S[i+1].y-S[i].y});
        return vnorm(Vec2{S[i].x-S[i-1].x, S[i].y-S[i-1].y});
    };
    auto local_dir_closed = [&](const vector<Vec2>&S, size_t i)->Vec2{
        if(S.size()<2) return {1,0}; size_t j=(i+1)%S.size(); return vnorm(Vec2{S[j].x-S[i].x, S[j].y-S[i].y});
    };
    auto rotate_both = [&](vector<Vec2>&A, vector<int>&B, size_t k){
        std::rotate(A.begin(), A.begin()+k, A.end());
        std::rotate(B.begin(), B.begin()+k, B.end());
    };

    if (!closed_mode){
        // OPEN: 방향=현재 차량 헤딩과 정렬(현재 위치에 가장 가까운 중점의 국소 방향 사용), 이후 시작점=초기 위치 가장 가까운 중점으로 회전
        Vec2 curr_pos{C.current_pos_x, C.current_pos_y};
        Vec2 curr_dir=vnorm(dir_from_heading_rad(C.current_heading_rad));
        size_t i_near_cur = nearest_idx(R.ordered, curr_pos);
        Vec2 vloc = local_dir_open(R.ordered, i_near_cur);
        if (geom::dot(vloc, curr_dir) < 0.0) reverse_both(R.ordered, R.mids_order_idx);

        Vec2 start_anchor{C.start_anchor_x, C.start_anchor_y};
        size_t k = nearest_idx(R.ordered, start_anchor);
        rotate_both(R.ordered, R.mids_order_idx, k);
    } else {
        // CLOSED:
        // 초기 헤딩 기준 방향 정합 (기본)
        Vec2 start_anchor{C.start_anchor_x, C.start_anchor_y};
        Vec2 init_dir = vnorm(dir_from_heading_rad(C.start_heading_rad));
        size_t i_near_anchor = nearest_idx(R.ordered, start_anchor);
        Vec2 vloc = local_dir_closed(R.ordered, i_near_anchor);
        if (!cfg::get().force_closed_ccw && geom::dot(vloc, init_dir) < 0.0)
            reverse_both(R.ordered, R.mids_order_idx);

        // 시작점 회전: start_anchor에 가장 가까운 중점이 index 0
        i_near_anchor = nearest_idx(R.ordered, start_anchor);
        rotate_both(R.ordered, R.mids_order_idx, i_near_anchor);
    }

    return R;
}

struct ReconstructedRings{
    vector<Vec2> inner_from_mids, outer_from_mids;
};

// ordered mids의 진행방향 기준으로 inner/outer도 동일한 진행방향으로 맞춤
static ReconstructedRings reconstruct_rings_and_align(
    const OrderedMids& OM,
    const MidsFiltered& MF,
    const Triangulation& T,
    const string& base)
{
    ReconstructedRings R;
    vector<char> seen_inner(T.all.size(),0), seen_outer(T.all.size(),0);

    auto get_io = [&](int u,int v,int&iv,int&ov){
        if (T.label[u]==0 && T.label[v]==1){ iv=u; ov=v; }
        else if (T.label[u]==1 && T.label[v]==0){ iv=v; ov=u; }
        else { iv=ov=-1; }
    };

    for (int k=0;k<(int)OM.mids_order_idx.size();++k){
        int mid_local = OM.mids_order_idx[k];    // mids[] 기준
        int eidx = MF.keep_edge_idx[mid_local];  // binfo 인덱스
        int u = MF.binfo[eidx].u, v=MF.binfo[eidx].v;
        int iv=-1, ov=-1; get_io(u,v,iv,ov);
        if (iv>=0 && !seen_inner[iv]){ R.inner_from_mids.push_back(T.all[iv]); seen_inner[iv]=1; }
        if (ov>=0 && !seen_outer[ov]){ R.outer_from_mids.push_back(T.all[ov]); seen_outer[ov]=1; }
    }

    auto vnorm=[&](const Vec2&a)->Vec2{ return geom::normalize(a,1e-12); };
    auto first_dir=[&](const vector<Vec2>&S)->Vec2{
        if(S.size()<2) return Vec2{1,0};
        return vnorm(Vec2{S[1].x-S[0].x, S[1].y-S[0].y});
    };
    Vec2 v_ref = first_dir(OM.ordered);
    auto align_to_mids_dir=[&](vector<Vec2>& seq){
        if(seq.size()<2) return;
        Vec2 v_seq = first_dir(seq);
        if (geom::dot(v_ref, v_seq) < 0.0) std::reverse(seq.begin(), seq.end());
    };
    align_to_mids_dir(R.inner_from_mids);
    align_to_mids_dir(R.outer_from_mids);

    io::saveCSV_pointsXY(base + "_inner_from_mids.csv", R.inner_from_mids);
    io::saveCSV_pointsXY(base + "_outer_from_mids.csv", R.outer_from_mids);

    if (cfg::get().verbose){
        int want_in=0,want_out=0; for(int i=0;i<(int)T.label.size();++i){ want_in+=(T.label[i]==0); want_out+=(T.label[i]==1); }
        cerr << "[cones-from-mids] inner used " << R.inner_from_mids.size() << "/" << want_in
             << ", outer used " << R.outer_from_mids.size() << "/" << want_out << "\n";
    }
    return R;
}

struct CenterlineOut{
    vector<Vec2> center;
    centerline::Spline1D spx, spy;
    double s0=0.0, L=0.0;
};

static int dynamic_samples_from_mids_count(int mids_n){
    auto &C = cfg::get();
    int dyn = (int)std::llround(C.sample_factor_n * std::max(0, mids_n));
    dyn = std::max(dyn, C.samples_min);
    if (C.samples_max > 0) dyn = std::min(dyn, C.samples_max);
    dyn = std::max(dyn, 4);
    return dyn;
}

static CenterlineOut make_centerline(const OrderedMids& OM, bool closed_mode, const string& base){
    auto &C = cfg::get();
    centerline::Spline1D spx,spy; double s0=0,L=0;
    int paddingK = closed_mode ? 3 : 0;
    vector<Vec2> center = centerline::splineUniformResample(
        OM.ordered, C.samples, paddingK, /*close_loop=*/C.emit_closed_duplicate, spx,spy,s0,L);

    io::saveCSV_pointsXY(base + "_mids_raw.csv", OM.ordered); // 참고용(ordered만 저장)
    return {center, std::move(spx), std::move(spy), s0, L};
}

static void save_centerline_csv(const string& outPath, const vector<Vec2>& center){
    std::ofstream fo(outPath);
    if (!fo) throw std::runtime_error("save centerline failed: " + outPath);
    fo.setf(std::ios::fixed); fo.precision(9);
    for (auto&p: center) fo<<p.x<<","<<p.y<<"\n";
}

static void compute_geom_and_save(const string& base,
                                  const vector<Vec2>& center,
                                  const centerline::Spline1D& spx, const centerline::Spline1D& spy,
                                  double s0, double L, bool closed_mode,
                                  const vector<Vec2>& inner_from_mids,
                                  const vector<Vec2>& outer_from_mids)
{
    auto &C = cfg::get();
    vector<pair<Vec2,Vec2>> innerE = closed_mode ? edges::ringEdges(inner_from_mids)
                                                 : edges::polylineEdges(inner_from_mids);
    vector<pair<Vec2,Vec2>> outerE = closed_mode ? edges::ringEdges(outer_from_mids)
                                                 : edges::polylineEdges(outer_from_mids);

    std::ofstream fo2(base + "_with_geom.csv");
    if (!fo2) throw std::runtime_error("save centerline_with_geom failed");
    fo2.setf(std::ios::fixed); fo2.precision(9);
    fo2 << "s,x,y,heading_rad,curvature,dist_to_inner,dist_to_outer,width,v_kappa_mps\n";

    double x0=0,y0=0,hd0=0,k0=0,din0=0,dout0=0,w0=0,v0=0;
    const int Ncenter=(int)center.size();
    const int Kmax  = closed_mode ? C.samples : Ncenter;
    const int denomN= closed_mode ? C.samples : std::max(1, C.samples);

    for(int k=0;k<Kmax;++k){
        double si = s0 + L * (double(k) / double(denomN));
        double x,xp,xpp,y,yp,ypp; spx.eval_with_deriv(si,x,xp,xpp); spy.eval_with_deriv(si,y,yp,ypp);
        double heading=std::atan2(yp,xp);
        double speed2=xp*xp+yp*yp; double denom=std::pow(std::max(1e-12, speed2),1.5);
        double curv=(xp*ypp - yp*xpp)/denom;

        geom::Vec2 nvec = geom::normalize(geom::Vec2{-yp, xp}, 1e-12);
        double d_in=0.0, d_out=0.0;
        if(nvec.x!=0 || nvec.y!=0) distancesToRings({x,y}, nvec, innerE, outerE, d_in, d_out);
        double width=d_in+d_out;
        double denom_k=std::max(std::fabs(curv), C.kappa_eps);
        double v_kappa=std::sqrt(C.a_lat_max/denom_k); if(v_kappa>C.v_cap_mps) v_kappa=C.v_cap_mps;

        if(k==0){ x0=x; y0=y; hd0=heading; k0=curv; din0=d_in; dout0=d_out; w0=width; v0=v_kappa; }

        double si_rel=si - s0;
        fo2 << si_rel << "," << x << "," << y << "," << heading << "," << curv
            << "," << d_in << "," << d_out << "," << width << "," << v_kappa << "\n";
    }
    if (C.emit_closed_duplicate){
        fo2 << L << "," << x0 << "," << y0 << "," << hd0 << "," << k0
            << "," << din0 << "," << dout0 << "," << w0 << "," << v0 << "\n";
    }
}

static void compute_raceline_and_save(const string& base,
                                      const vector<Vec2>& center_for_opt, double s0, double L, bool closed_mode,
                                      const vector<Vec2>& inner_from_mids, const vector<Vec2>& outer_from_mids)
{
    auto &C = cfg::get();
    vector<pair<Vec2,Vec2>> innerE = closed_mode ? edges::ringEdges(inner_from_mids)
                                                 : edges::polylineEdges(inner_from_mids);
    vector<pair<Vec2,Vec2>> outerE = closed_mode ? edges::ringEdges(outer_from_mids)
                                                 : edges::polylineEdges(outer_from_mids);

    auto res = raceline_min_curv::compute_min_curvature_raceline(
        center_for_opt, innerE, outerE, C.veh_width_m, L, closed_mode);

    // 좌표 저장
    {
        std::ofstream fo(base + "_raceline.csv");
        if(!fo) throw std::runtime_error("save raceline failed");
        fo.setf(std::ios::fixed); fo.precision(9);
        for(auto&p: res.raceline) fo<<p.x<<","<<p.y<<"\n";
        if(C.emit_closed_duplicate && !res.raceline.empty())
            fo<<res.raceline[0].x<<","<<res.raceline[0].y<<"\n";
    }
    // with geom
    {
        std::ofstream fo(base + "_raceline_with_geom.csv");
        if(!fo) throw std::runtime_error("save raceline_with_geom failed");
        fo.setf(std::ios::fixed); fo.precision(9);
        fo << "s,x,y,heading_rad,curvature,alpha_last,v_kappa_mps\n";
        int Nrl=(int)res.raceline.size();
        for(int k=0;k<Nrl;++k){
            double si = s0 + L * (double(k) / double(std::max(1,Nrl)));
            double si_rel = si - s0;
            double denom_k = std::max(std::fabs(res.curvature[k]), C.kappa_eps);
            double v_kappa = std::sqrt(C.a_lat_max/denom_k); if(v_kappa>C.v_cap_mps) v_kappa=C.v_cap_mps;
            fo << si_rel << "," << res.raceline[k].x << "," << res.raceline[k].y << ","
               << res.heading[k] << "," << res.curvature[k] << "," << res.alpha_last[k] << ","
               << v_kappa << "\n";
        }
        if (C.emit_closed_duplicate && !res.raceline.empty()){
            double denom_k0 = std::max(std::fabs(res.curvature[0]), C.kappa_eps);
            double v0 = std::sqrt(C.a_lat_max/denom_k0); if(v0>C.v_cap_mps) v0=C.v_cap_mps;
            fo << L << "," << res.raceline[0].x << "," << res.raceline[0].y << ","
               << res.heading[0] << "," << res.curvature[0] << "," << res.alpha_last[0] << ","
               << v0 << "\n";
        }
    }
}

static void compute_mintime_and_save(const string &base,
                                     const vector<Vec2> &center_for_opt,
                                     double s0, double L, bool closed_mode,
                                     const vector<Vec2> &inner_from_mids,
                                     const vector<Vec2> &outer_from_mids)
{
    auto &C = cfg::get();
    vector<pair<Vec2, Vec2>> innerE = closed_mode ? edges::ringEdges(inner_from_mids)
                                                  : edges::polylineEdges(inner_from_mids);
    vector<pair<Vec2, Vec2>> outerE = closed_mode ? edges::ringEdges(outer_from_mids)
                                                  : edges::polylineEdges(outer_from_mids);

    auto res = raceline_min_time::compute_min_time_raceline(
        center_for_opt, innerE, outerE, C.veh_width_m, L, closed_mode);

    // 좌표
    {
        std::ofstream fo(base + "_mintime_raceline.csv");
        if (!fo)
            throw std::runtime_error("save mintime_raceline failed");
        fo.setf(std::ios::fixed);
        fo.precision(9);
        for (auto &p : res.raceline)
            fo << p.x << "," << p.y << "\n";
        if (C.emit_closed_duplicate && !res.raceline.empty())
            fo << res.raceline[0].x << "," << res.raceline[0].y << "\n";
    }

    // 기하+속도/가속 / 랩타임
    {
        std::ofstream fo(base + "_mintime_with_geom.csv");
        if (!fo)
            throw std::runtime_error("save mintime_with_geom failed");
        fo.setf(std::ios::fixed);
        fo.precision(9);
        fo << "s,x,y,heading_rad,curvature,alpha_last,v_mps,ax_mps2\n";
        int N = (int)res.raceline.size();
        for (int k = 0; k < N; ++k)
        {
            double si = s0 + L * (double(k) / double(std::max(1, N)));
            double si_rel = si - s0;
            fo << si_rel << "," << res.raceline[k].x << "," << res.raceline[k].y << ","
               << res.heading[k] << "," << res.curvature[k] << "," << res.alpha_last[k] << ","
               << res.v[k] << "," << res.ax[k] << "\n";
        }
        if (C.emit_closed_duplicate && N > 0)
        {
            fo << L << "," << res.raceline[0].x << "," << res.raceline[0].y << ","
               << res.heading[0] << "," << res.curvature[0] << "," << res.alpha_last[0] << ","
               << res.v[0] << "," << res.ax[0] << "\n";
        }
    }
    std::cerr << "[mintime] Estimated laptime: " << std::fixed << std::setprecision(3)
              << res.lap_time << " s\n";

    // ====================== DEBUG: center / min-curv / min-time 비교 ======================
    if (cfg::get().debug_dump) {
        auto &C = cfg::get();

        // 1) 보조 유틸
        auto path_length = [&](const vector<Vec2>& P, bool closed)->double{
            double Ltot=0.0; int Np=(int)P.size(); if(Np<=1) return 0.0;
            for(int i=0;i+1<Np;++i){ Ltot += std::hypot(P[i+1].x-P[i].x, P[i+1].y-P[i].y); }
            if (closed && Np>=2) Ltot += std::hypot(P[0].x-P[Np-1].x, P[0].y-P[Np-1].y);
            return Ltot;
        };
        auto clamp01 = [&](double x){ return x<0?0:(x>1?1:x); };

        // 2) (선택) 최소곡률 경로 로드
        vector<Vec2> raceline_mincurv = io::loadCSV_XY(base + "_raceline.csv");
        if (!raceline_mincurv.empty() && closed_mode &&
            raceline_mincurv.size()>=2 &&
            geom::almostEq(raceline_mincurv.front(), raceline_mincurv.back(), 1e-12))
        {
            raceline_mincurv.pop_back();
        }

        // 3) 센터라인 법선(부호 있는 편차 산출용)
        vector<Vec2> n_center;
        raceline_min_curv::normals_from_points_generic(center_for_opt, closed_mode, n_center);

        // 4) 센터라인/최소곡률의 랩타임(동일 동역학으로) 산출해서 참고치 출력
        auto Hc = center_for_opt.size()>0 ? (L / std::max(1,(int)center_for_opt.size())) : 1.0;
        vector<double> hd_c, kap_c;
        raceline_min_curv::heading_curv_from_points_generic(center_for_opt, Hc, closed_mode, hd_c, kap_c);
        auto VP_c = raceline_min_time::velocity_profile_forward_backward(kap_c, Hc, closed_mode);

        double lap_mc = -1.0;
        if (!raceline_mincurv.empty()){
            double L_mc = path_length(raceline_mincurv, closed_mode);
            double Hmc  = L_mc / std::max(1,(int)raceline_mincurv.size());
            vector<double> hd_mc, kap_mc;
            raceline_min_curv::heading_curv_from_points_generic(raceline_mincurv, Hmc, closed_mode, hd_mc, kap_mc);
            auto VP_mc = raceline_min_time::velocity_profile_forward_backward(kap_mc, Hmc, closed_mode);
            lap_mc = VP_mc.lap_time;
            std::cerr << "[debug] centerline lap ≈ " << std::setprecision(3) << VP_c.lap_time
                    << " s,  min-curv lap ≈ " << VP_mc.lap_time
                    << " s,  min-time lap ≈ " << res.lap_time << " s\n";
        } else {
            std::cerr << "[debug] centerline lap ≈ " << std::setprecision(3) << VP_c.lap_time
                    << " s,  (min-curv not found),  min-time lap ≈ " << res.lap_time << " s\n";
        }

        // 5) 비교 CSV 덤프
        const int N = (int)std::min<size_t>(res.raceline.size(), center_for_opt.size());
        std::ofstream fcmp(base + "_debug_compare_paths.csv");
        if (fcmp){
            fcmp.setf(std::ios::fixed); fcmp.precision(9);
            fcmp << "s,cx,cy,mt_x,mt_y,mc_x,mc_y,d_mt_signed_m,d_mc_signed_m,"
                    "d_mt_abs_m,d_mc_abs_m,kappa_mt,v_mt,ax_mt,alat_mt,alat_ratio,"
                    "gamma,a_acc_cap,a_brk_cap,a_power_cap\n";

            // 통계
            double sum_abs_mt=0.0, sum2_abs_mt=0.0, max_abs_mt=0.0; int idx_max_mt=0;
            double sum_abs_mc=0.0, sum2_abs_mc=0.0, max_abs_mc=0.0; int idx_max_mc=0;
            int near_cnt = 0;

            // 전 구간 루프
            for(int k=0;k<N;++k){
                double si = s0 + L * (double(k) / double(std::max(1,N)));
                const Vec2& Cpt = center_for_opt[k];
                const Vec2& Mt  = res.raceline[k];
                Vec2 Mc {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
                if ((int)raceline_mincurv.size()>k) Mc = raceline_mincurv[k];

                // 부호 있는 편차 (센터라인 법선 기준)
                auto dmt = (Mt.x - Cpt.x)*n_center[k].x + (Mt.y - Cpt.y)*n_center[k].y;
                auto dmc = (std::isfinite(Mc.x) ? (Mc.x - Cpt.x)*n_center[k].x + (Mc.y - Cpt.y)*n_center[k].y
                                                : std::numeric_limits<double>::quiet_NaN());
                double admt = std::fabs(dmt);
                double admc = std::isfinite(dmc)? std::fabs(dmc) : std::numeric_limits<double>::quiet_NaN();

                // 사용된 횡가속/가중치
                double kappa_mt = (k<(int)res.curvature.size()? res.curvature[k]:0.0);
                double v_mt     = (k<(int)res.v.size()?         res.v[k]        :0.0);
                double ax_mt    = (k<(int)res.ax.size()?        res.ax[k]       :0.0);
                double alat     = v_mt*v_mt*std::fabs(kappa_mt);
                double alat_ratio = (C.a_total_max>1e-9)? std::min(1.0, alat / C.a_total_max) : 0.0;

                double vkappa = std::sqrt(C.a_lat_max / std::max(std::fabs(kappa_mt), C.kappa_eps));
                double pen = clamp01( (vkappa - v_mt) / std::max(1e-6, vkappa) );
                double gamma = 1.0 + C.w_time_gain * pen;

                // 종가속 한계(설명용)
                auto ax_caps = [&](double vi, double ki){
                    double alatloc = vi*vi*std::fabs(ki);
                    double a_res = std::sqrt(std::max(0.0, C.a_total_max*C.a_total_max - alatloc*alatloc));
                    double Fd = 0.5*C.rho_air*C.Cd*C.A_front_m2*vi*vi;
                    double Fr = C.mass_kg*9.81*C.c_rr;
                    double a_power = (C.P_max_W>0 && vi>1e-6)? (C.P_max_W/(C.mass_kg*vi) - (Fd+Fr)/C.mass_kg) : 1e9;
                    double a_acc_cap = std::max(0.0, std::min({a_res, C.a_long_acc_cap, a_power}));
                    double a_brk_cap = std::max(0.0, std::min(a_res, C.a_long_brake_cap) + (Fd+Fr)/C.mass_kg);
                    return std::tuple<double,double,double>(a_acc_cap, a_brk_cap, std::max(0.0, a_power));
                };
                double a_acc_cap, a_brk_cap, a_power_cap;
                std::tie(a_acc_cap,a_brk_cap,a_power_cap) = ax_caps(v_mt, kappa_mt);

                // CSV 기록
                fcmp << (si - s0) << ","
                    << Cpt.x << "," << Cpt.y << ","
                    << Mt.x  << "," << Mt.y  << ","
                    << Mc.x  << "," << Mc.y  << ","
                    << dmt   << "," << dmc   << ","
                    << admt  << "," << admc  << ","
                    << kappa_mt << "," << v_mt << "," << ax_mt << ","
                    << alat << "," << alat_ratio << ","
                    << gamma << "," << a_acc_cap << "," << a_brk_cap << "," << a_power_cap << "\n";

                // 통계 집계
                sum_abs_mt   += admt; sum2_abs_mt += admt*admt;
                if (admt > max_abs_mt){ max_abs_mt = admt; idx_max_mt = k; }
                if (std::isfinite(admc)){
                    sum_abs_mc   += admc; sum2_abs_mc += admc*admc;
                    if (admc > max_abs_mc){ max_abs_mc = admc; idx_max_mc = k; }
                }
                if (admt < C.debug_offset_warn_m) near_cnt++;
            }
            fcmp.close();

            // 6) 요약 로그
            double mean_mt = sum_abs_mt / std::max(1,N);
            double rms_mt  = std::sqrt(sum2_abs_mt / std::max(1,N));
            std::cerr << std::setprecision(3)
                    << "[debug] min-time vs center: mean|offset|=" << mean_mt
                    << " m, rms=" << rms_mt
                    << " m, max=" << max_abs_mt << " m @i=" << idx_max_mt
                    << ", within " << C.debug_offset_warn_m << " m : "
                    << near_cnt << "/" << N << "\n";
            if (max_abs_mc>0.0 && std::isfinite(max_abs_mc)){
                double mean_mc = sum_abs_mc / std::max(1,N);
                double rms_mc  = std::sqrt(sum2_abs_mc / std::max(1,N));
                std::cerr << "[debug] min-curv vs center: mean|offset|=" << mean_mc
                        << " m, rms=" << rms_mc
                        << " m, max=" << max_abs_mc << " m @i=" << idx_max_mc << "\n";
            }

            // 7) “센터라인과 너무 비슷한가?” 판단 힌트
            if (mean_mt < 0.01 && max_abs_mt < 0.03){
                std::cerr << "[hint] min-time 경로가 센터라인과 매우 유사합니다. "
                            "폭이 좁거나 곡률/동역학 제약이 강해서 경로 이동 여지가 작을 수 있어요.\n";
            }
            if (lap_mc>0.0){
                double gain_vs_mc = (lap_mc - res.lap_time) / lap_mc * 100.0;
                std::cerr << "[debug] lap gain vs min-curv: " << gain_vs_mc << " %\n";
            }
        } else {
            std::cerr << "[debug] cfg::debug_dump=false (비교 CSV 생략)\n";
        }
    }
}
}// namespace pipeline

// ================================== MAIN ==================================
int main(int argc, char** argv){
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (argc < 4){
        std::cerr << "Usage: " << argv[0] << " inner.csv outer.csv centerline.csv\n";
        return 1;
    }
    const string innerPath = argv[1], outerPath = argv[2], outPath = argv[3];
    const string base = io::dropExt(outPath);

    auto &C = cfg::get();

    double T_load=0, T_dt=0, T_mids=0, T_orderMST=0, T_spline=0, T_saveC=0, T_geom=0, T_mincurv_race=0, T_mintime_race=0, T_saveR=0;

    // 1) 입력
    vector<geom::Vec2> inner, outer;
    {
        timing::ScopedAcc _t("1) 입력 로드", &T_load);
        inner = io::loadCSV_XY(innerPath);
        outer = io::loadCSV_XY(outerPath);
    }
    if (inner.size()<2 || outer.size()<2){ std::cerr<<"[ERR] need >=2 points per ring\n"; return 2; }
    const bool closed_mode = C.is_closed_track;

    // 2) DT
    pipeline::Triangulation tri;
    {
        timing::ScopedAcc _t("2) Delaunay(BW)", &T_dt);
        tri = pipeline::buildDT(inner, outer);
        if (C.verbose) std::cerr << "[DT] points="<<tri.all.size()<<" faces="<<tri.tris.size()<<"\n";
        io::saveCSV_pointsLabeled(base + "_all_points.csv", tri.all, tri.label);
        io::saveCSV_trisIdx(base + "_tri_raw_idx.csv", tri.tris);
    }

    // 3) 경계 엣지 중점 + 길이 필터
    pipeline::MidsFiltered MF;
    {
        timing::ScopedAcc _t("3) 경계엣지 중점+길이필터", &T_mids);
        MF = pipeline::extract_mids_with_len_filter(tri, base);
        if (C.use_dynamic_samples){
            C.samples = pipeline::dynamic_samples_from_mids_count((int)MF.mids.size());
            if (C.verbose) std::cerr<<"[samples] dynamic="<<C.samples<<" (n="<<C.sample_factor_n<<", mids="<<MF.mids.size()<<")\n";
        }
    }

    // 4) MST 순서화 + 방향/시작점 정합
    pipeline::OrderedMids OM;
    {
        timing::ScopedAcc _t("4) 중점 순서화(MST)+방향/시작점 정합", &T_orderMST);
        OM = pipeline::order_and_align_mids_open_closed(MF.mids, closed_mode);
        io::saveCSV_pointsXY(base + "_mids_ordered.csv", OM.ordered);
    }

    // 5) 중점 순서를 이용해 링 재구성 + 방향 정렬
    pipeline::ReconstructedRings RR;
    {
        RR = pipeline::reconstruct_rings_and_align(OM, MF, tri, base);
    }

    // 6) 스플라인 + 균일 재샘플 → centerline
    pipeline::CenterlineOut CL;
    {
        timing::ScopedAcc _t("5) 스플라인+균일 재샘플", &T_spline);
        CL = pipeline::make_centerline(OM, closed_mode, base);
    }

    // 7) centerline 저장
    {
        timing::ScopedAcc _t("6) centerline 저장", &T_saveC);
        pipeline::save_centerline_csv(outPath, CL.center);
    }

    // 8) geom/width 계산 및 저장
    {
        timing::ScopedAcc _t("7) geom+width 계산/저장", &T_geom);
        pipeline::compute_geom_and_save(base, CL.center, CL.spx, CL.spy, CL.s0, CL.L, closed_mode,
                                        RR.inner_from_mids, RR.outer_from_mids);
    }

    // 9) 최소곡률 레이싱라인 최적화 & 저장
    {
        timing::ScopedAcc _t("8) 최소곡률 레이싱라인 최적화+저장", &T_mincurv_race);
        vector<geom::Vec2> center_for_opt = CL.center;
        if (closed_mode && center_for_opt.size()>=2 && geom::almostEq(center_for_opt.front(), center_for_opt.back(), 1e-12))
            center_for_opt.pop_back();
        pipeline::compute_raceline_and_save(base, center_for_opt, CL.s0, CL.L, closed_mode,
                                            RR.inner_from_mids, RR.outer_from_mids);
    }

    // 10) 최소시간 레이싱라인 최적화 & 저장
    {
        timing::ScopedAcc _t("9) 최소시간 레이싱라인 최적화+저장", &T_mintime_race);
        vector<geom::Vec2> center_for_opt = CL.center;
        if (closed_mode && center_for_opt.size()>=2 && geom::almostEq(center_for_opt.front(), center_for_opt.back(), 1e-12))
            center_for_opt.pop_back();
        pipeline::compute_mintime_and_save(base, center_for_opt, CL.s0, CL.L, closed_mode,
                                        RR.inner_from_mids, RR.outer_from_mids);
    }

    // 요약
    {
        std::cerr.setf(std::ios::fixed);
        std::cerr << "[TIME][SUMMARY] "
                  << "load="<<T_load
                  << ", dt="<<T_dt
                  << ", mids="<<T_mids
                  << ", mst="<<T_orderMST
                  << ", spline="<<T_spline
                  << ", saveC="<<T_saveC
                  << ", geom="<<T_geom
                  << ", mincurv_race="<<T_mincurv_race
                  << ", mintime_race="<<T_mintime_race
                  << "\n";
    }
    return 0;
}
