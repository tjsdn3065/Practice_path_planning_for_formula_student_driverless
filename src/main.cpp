// main.cpp
// ============================================================================
// 트랙 센터라인 + 폭 계산(폐루프/비폐루프 모두) + 최소 곡률 레이싱라인
//  - is_closed_track=true/false 로 알고리즘 모드 선택
//  - emit_closed_duplicate=true  → 출력 CSV에 마지막 샘플을 첫 샘플로 복제
// ---------------------------------------------------------------------------
// Pipeline (DT-only):
//  1) 입력 링(inner/outer) -> Delaunay(Bowyer–Watson)
//  2) 내/외부 라벨 경계 엣지의 중점 추출(+길이 필터)
//  3) MST 지름 경로 기반 중점 순서화 + 방향/시작점 정합(OPEN/폐곡 규칙)
//  4) 자연 3차 스플라인(TDMA) + 경계 패딩 -> 균일 arc-length 재샘플
//  5) 각 샘플점에서 법선으로 inner/outer까지 거리 -> 코리도 폭 w_L, w_R
//  6) 최소 곡률(∑κ^2 + λ||D1α||^2) 최소화,  lo ≤ α ≤ hi  (α: 법선 오프셋)
//     - GN(선형화) - projected step - outer relinearization
//  7) 결과 저장 (centerline.csv, *_with_geom.csv, *_raceline*.csv)
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

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

using std::array;
using std::cerr;
using std::cout;
using std::endl;
using std::pair;
using std::string;
using std::vector;

//================================= Config =================================
namespace cfg
{
    struct Config
    {
        // --- Anchors for orientation/rotation ---
        double current_pos_x = 4.994457593, current_pos_y = -0.108866966; // 차량 현재 위치(오픈 트랙용)
        // 헤딩 라디안: +x가 0rad, CCW가 + 방향 (표준 수학 좌표계)
        double current_heading_rad = 0.046698693;
        double start_anchor_x = 0.0, start_anchor_y = 0.0; // 초기(시작) 위치(폐루프 회전/정렬용)
        double start_heading_rad = 0.0;                     // 초기(시작) 헤딩(폐루프 방향 정합용)

        // 알고리즘 모드(트랙 폐루프 여부)
        bool is_closed_track = true;

        // 출력 옵션: 폐루프 시 마지막 샘플 = 첫 샘플 복제 여부
        bool emit_closed_duplicate = true;

        // 전처리/샘플링
        bool add_micro_jitter = true;
        double jitter_eps = 1e-9;

        int samples = 300;               // 센터라인 재샘플 수
        bool use_dynamic_samples = true; // true면 콘 개수 기준 자동 설정
        double sample_factor_n = 1.5;    // n: 샘플 수 = n * (#mids)
        int samples_min = 10;            // 과도한 희소화 방지
        int samples_max = 500;           // 과샘플 방지

        int knn_k = 8; // MST k-NN

        // 길이 필터
        bool enable_boundary_len_filter = true;
        double boundary_edge_len_scale = 2.0; // 길이 컷오프 = scale * 중앙값
        double boundary_edge_abs_max = 6.0;   // 절대 상한(미사용하려면 크게)

        // 차량 폭/마진 (코리도 가드)
        double veh_width_m = 1.0;
        double safety_margin_m = 0.05;

        // --- 논문식 raceline 최적화 파라미터 ---
        double lambda_smooth = 1e-3; // 스무딩 가중치 λ (||D1 α||^2)
        int max_outer_iters = 16;    // GN 재선형화 횟수
        int max_inner_iters = 150;   // 내부 projected step
        double step_init = 0.6;      // 초기 스텝 (Armijo)
        double step_min = 1e-6;      // 최소 스텝
        double armijo_c = 1e-5;      // Armijo 계수

        double a_lat_max = 10.0;   // A_LAT_MAX
        double kappa_eps = 1e-6;   // KAPPA_EPS
        double v_cap_mps = 27.0;   // 최대 속도 캡
    };
    inline Config &get()
    {
        static Config C;
        return C;
    }
} // namespace cfg

//============================== Geometry ===================================
namespace geom
{
    struct Vec2
    {
        double x = 0.0, y = 0.0;
    };
    inline Vec2 operator+(const Vec2 &a, const Vec2 &b) { return {a.x + b.x, a.y + b.y}; }
    inline Vec2 operator-(const Vec2 &a, const Vec2 &b) { return {a.x - b.x, a.y - b.y}; }
    inline Vec2 operator*(const Vec2 &a, double s) { return {a.x * s, a.y * s}; }

    inline double dot(const Vec2 &a, const Vec2 &b) { return a.x * b.x + a.y * b.y; }
    inline double norm2(const Vec2 &a) { return dot(a, a); }
    inline double norm(const Vec2 &a) { return std::sqrt(norm2(a)); }

    inline Vec2 normalize(const Vec2 &v, double eps = 1e-12)
    {
        double n = norm(v);
        if (n < eps)
            return {0, 0};
        return {v.x / n, v.y / n};
    }

    inline bool almostEq(const Vec2 &a, const Vec2 &b, double e = 1e-12)
    {
        return (std::fabs(a.x - b.x) <= e && std::fabs(a.y - b.y) <= e);
    }

    // robust orient/incircle
    inline double orient2d_filt(const Vec2 &a, const Vec2 &b, const Vec2 &c)
    {
        double det = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        double absa = std::fabs(b.x - a.x) + std::fabs(b.y - a.y);
        double absb = std::fabs(c.x - a.x) + std::fabs(c.y - a.y);
        double err = (absa * absb) * std::numeric_limits<double>::epsilon() * 4.0;
        if (std::fabs(det) > err)
            return det;
        long double adx = (long double)b.x - (long double)a.x;
        long double ady = (long double)b.y - (long double)a.y;
        long double bdx = (long double)c.x - (long double)a.x;
        long double bdy = (long double)c.y - (long double)a.y;
        long double detl = adx * bdy - ady * bdx;
        return (double)detl;
    }
    inline double incircle_filt(const Vec2 &a, const Vec2 &b, const Vec2 &c, const Vec2 &d)
    {
        double adx = a.x - d.x, ady = a.y - d.y;
        double bdx = b.x - d.x, bdy = b.y - d.y;
        double cdx = c.x - d.x, cdy = c.y - d.y;
        double ad = adx * adx + ady * ady, bd = bdx * bdx + bdy * bdy, cd = cdx * cdx + cdy * cdy;
        double det = adx * (bdy * cd - bd * cdy) - ady * (bdx * cd - bd * cdx) + ad * (bdx * cdy - bdy * cdx);
        double mags = (std::fabs(adx) + std::fabs(ady)) * (std::fabs(bdx) + std::fabs(bdy)) * (std::fabs(cdx) + std::fabs(cdy));
        double err = mags * std::numeric_limits<double>::epsilon() * 16.0;
        if (std::fabs(det) > err)
            return det;
        long double AX = a.x, AY = a.y, BX = b.x, BY = b.y, CX = c.x, CY = c.y, DX = d.x, DY = d.y;
        long double adxl = AX - DX, adyl = AY - DY, bdxl = BX - DX, bdyl = BY - DY, cdxl = CX - DX, cdyl = CY - DY;
        long double adl = adxl * adxl + adyl * adyl, bdl = bdxl * bdxl + bdyl * bdyl, cdl = cdxl * cdxl + cdyl * cdyl;
        long double detl = adxl * (bdyl * cdl - bdl * cdyl) - adyl * (bdxl * cdl - bdl * cdxl) + adl * (bdxl * cdyl - bdyl * cdxl);
        return (double)detl;
    }

    inline bool ccw(const Vec2 &A, const Vec2 &B, const Vec2 &C) { return orient2d_filt(A, B, C) > 0; }

} // namespace geom

//============================== Delaunay / CDT =============================
namespace delaunay
{
    using geom::Vec2;

    struct Tri
    {
        int a, b, c;
    }; // CCW

    // Bowyer–Watson
    static vector<Tri> bowyerWatson(const vector<Vec2> &pts)
    {
        auto &C = cfg::get();
        vector<Vec2> P = pts;
        if (C.add_micro_jitter)
        {
            std::mt19937_64 rng(1234567);
            std::uniform_real_distribution<double> U(-C.jitter_eps, C.jitter_eps);
            for (auto &p : P)
            {
                p.x += U(rng);
                p.y += U(rng);
            }
        }
        geom::Vec2 lo{+1e300, +1e300}, hi{-1e300, -1e300};
        for (const auto &p : P)
        {
            lo.x = std::min(lo.x, p.x);
            lo.y = std::min(lo.y, p.y);
            hi.x = std::max(hi.x, p.x);
            hi.y = std::max(hi.y, p.y);
        }
        geom::Vec2 c = (lo + hi) * 0.5;
        double d = std::max(hi.x - lo.x, hi.y - lo.y) * 1000.0 + 1.0;
        int n0 = (int)P.size();
        P.push_back({c.x - 2 * d, c.y - d});
        P.push_back({c.x + 2 * d, c.y - d});
        P.push_back({c.x, c.y + 2 * d});
        int s1 = n0, s2 = n0 + 1, s3 = n0 + 2;

        vector<Tri> T;
        T.push_back({s1, s2, s3});

        for (int ip = 0; ip < n0; ++ip)
        {
            const Vec2 &p = P[ip];
            vector<int> bad;
            bad.reserve(T.size() / 3);
            for (int t = 0; t < (int)T.size(); ++t)
            {
                auto &tr = T[t];
                if (!geom::ccw(P[tr.a], P[tr.b], P[tr.c]))
                    std::swap(tr.b, tr.c);
                if (geom::incircle_filt(P[tr.a], P[tr.b], P[tr.c], p) > 0)
                    bad.push_back(t);
            }
            struct E
            {
                int u, v;
            };
            vector<E> poly;
            auto addE = [&](int u, int v)
            {
                for (auto it = poly.begin(); it != poly.end(); ++it)
                {
                    if (it->u == v && it->v == u)
                    {
                        poly.erase(it);
                        return;
                    }
                }
                poly.push_back({u, v});
            };

            vector<char> del(T.size(), 0);
            for (int id : bad)
            {
                del[id] = 1;
                auto tr = T[id];
                addE(tr.a, tr.b);
                addE(tr.b, tr.c);
                addE(tr.c, tr.a);
            }
            vector<Tri> keep;
            keep.reserve(T.size());
            for (int i = 0; i < (int)T.size(); ++i)
                if (!del[i])
                    keep.push_back(T[i]);
            T.swap(keep);

            for (const auto &e : poly)
            {
                Tri nt{e.u, e.v, ip};
                if (!geom::ccw(P[nt.a], P[nt.b], P[nt.c]))
                    std::swap(nt.b, nt.c);
                T.push_back(nt);
            }
        }
        vector<Tri> out;
        out.reserve(T.size());
        for (const auto &tr : T)
        {
            if (tr.a >= n0 || tr.b >= n0 || tr.c >= n0)
                continue;
            out.push_back(tr);
        }
        return out;
    }

    struct EdgeKey
    {
        int u, v;
        EdgeKey() {}
        EdgeKey(int a, int b)
        {
            u = std::min(a, b);
            v = std::max(a, b);
        }
        bool operator==(const EdgeKey &o) const { return u == o.u && v == o.v; }
    };
    struct EdgeKeyHash
    {
        size_t operator()(const EdgeKey &k) const { return ((uint64_t)k.u << 32) ^ (uint64_t)k.v; }
    };
    struct EdgeRef
    {
        int tri;
        int a, b;
    };

    inline void buildEdgeMap(const vector<Tri> &T, std::unordered_map<EdgeKey, vector<EdgeRef>, EdgeKeyHash> &M)
    {
        M.clear();
        M.reserve(T.size() * 2);
        for (int t = 0; t < (int)T.size(); ++t)
        {
            const Tri &tr = T[t];
            int A[3] = {tr.a, tr.b, tr.c};
            for (int i = 0; i < 3; i++)
            {
                int u = A[i], v = A[(i + 1) % 3];
                M[EdgeKey(u, v)].push_back({t, u, v});
            }
        }
    }
} // namespace delaunay

//=========================== Clip & Simple Edges ===========================
namespace clip
{
    using geom::Vec2;

    inline vector<pair<Vec2, Vec2>> ringEdges(const vector<Vec2> &R)
    {
        vector<pair<Vec2, Vec2>> E;
        int n = (int)R.size();
        E.reserve(n);
        for (int i = 0; i < n; ++i)
        {
            int j = (i + 1) % n;
            E.push_back({R[i], R[j]});
        }
        return E;
    }
    inline vector<pair<Vec2, Vec2>> ringEdgesPolyline(const vector<Vec2> &R)
    {
        vector<pair<Vec2, Vec2>> E;
        int n = (int)R.size();
        if (n < 2)
            return E;
        E.reserve(n - 1);
        for (int i = 0; i + 1 < n; ++i)
            E.push_back({R[i], R[i + 1]});
        return E;
    }
} // namespace clip

//============================== Centerline =================================
namespace centerline
{
    using delaunay::Tri;
    using geom::Vec2;

    struct BoundaryEdgeInfo
    {
        int u, v;
        double len;
        bool is_hull;
        geom::Vec2 mid;
    };

    inline vector<BoundaryEdgeInfo>
    labelBoundaryEdges_with_len(const vector<geom::Vec2> &all,
                                const vector<delaunay::Tri> &T,
                                const vector<int> &labels)
    {
        using delaunay::buildEdgeMap;
        using delaunay::EdgeKey;
        using delaunay::EdgeKeyHash;
        using delaunay::EdgeRef;

        std::unordered_map<EdgeKey, vector<EdgeRef>, EdgeKeyHash> M;
        buildEdgeMap(T, M);

        vector<BoundaryEdgeInfo> out;
        out.reserve(M.size());
        for (const auto &kv : M)
        {
            int u = kv.first.u, v = kv.first.v;
            if (u < 0 || v < 0 || u >= (int)all.size() || v >= (int)all.size())
                continue;
            if (labels[u] < 0 || labels[v] < 0)
                continue;
            if (labels[u] == labels[v])
                continue;

            bool hull = (kv.second.size() == 1);
            double len = geom::norm(all[v] - all[u]);
            geom::Vec2 mid = (all[u] + all[v]) * 0.5;
            out.push_back({u, v, len, hull, mid});
        }
        return out;
    }

    inline vector<Vec2> orderByMST(const vector<Vec2> &pts)
    {
        auto &C = cfg::get();
        int n = (int)pts.size();
        if (n <= 2)
            return pts;

        int K = std::min(C.knn_k, n - 1);
        vector<vector<pair<int, double>>> adj(n);
        for (int i = 0; i < n; i++)
        {
            vector<pair<double, int>> cand;
            cand.reserve(n - 1);
            for (int j = 0; j < n; j++)
                if (i != j)
                {
                    double d2 = (pts[i].x - pts[j].x) * (pts[i].x - pts[j].x) + (pts[i].y - pts[j].y) * (pts[i].y - pts[j].y);
                    cand.push_back({d2, j});
                }
            if ((int)cand.size() > K)
            {
                std::nth_element(cand.begin(), cand.begin() + K, cand.end(),
                                 [](const auto &A, const auto &B)
                                 { return A.first < B.first; });
                cand.resize(K);
            }
            for (auto &c : cand)
            {
                double w = std::sqrt(std::max(0.0, c.first));
                adj[i].push_back({c.second, w});
                adj[c.second].push_back({i, w});
            }
        }

        vector<double> key(n, 1e300);
        vector<int> par(n, -1);
        vector<char> in(n, 0);
        key[0] = 0;
        for (int it = 0; it < n; ++it)
        {
            int u = -1;
            double best = 1e301;
            for (int i = 0; i < n; i++)
                if (!in[i] && key[i] < best)
                {
                    best = key[i];
                    u = i;
                }
            if (u == -1)
                break;
            in[u] = 1;
            for (auto [v, w] : adj[u])
                if (!in[v] && w < key[v])
                {
                    key[v] = w;
                    par[v] = u;
                }
        }

        vector<vector<int>> tree(n);
        for (int v = 0; v < n; v++)
            if (par[v] >= 0)
            {
                tree[v].push_back(par[v]);
                tree[par[v]].push_back(v);
            }

        auto bfs = [&](int s)
        {
            vector<double> d(n, 1e300);
            vector<int> p(n, -1);
            std::queue<int> q;
            q.push(s);
            d[s] = 0;
            while (!q.empty())
            {
                int u = q.front();
                q.pop();
                for (int v : tree[u])
                    if (d[v] > 1e299)
                    {
                        double w = std::sqrt((pts[u].x - pts[v].x) * (pts[u].x - pts[v].x) + (pts[u].y - pts[v].y) * (pts[u].y - pts[v].y));
                        d[v] = d[u] + w;
                        p[v] = u;
                        q.push(v);
                    }
            }
            int far = s;
            for (int i = 0; i < n; i++)
                if (d[i] > d[far])
                    far = i;
            return std::tuple<int, vector<int>, vector<double>>(far, p, d);
        };
        auto [s1, p1, d1] = bfs(0);
        auto [s2, p2, d2] = bfs(s1);

        vector<int> path;
        for (int v = s2; v != -1; v = p2[v])
            path.push_back(v);
        vector<char> used(n, 0);
        vector<Vec2> out;
        out.reserve(n);
        for (int id : path)
        {
            out.push_back(pts[id]);
            used[id] = 1;
        }
        for (int i = 0; i < n; i++)
            if (!used[i])
                out.push_back(pts[i]);
        return out;
    }

    // --------- 1D Natural Cubic Spline (TDMA) ----------
    struct Spline1D
    {
        vector<double> s, a, b, c, d;

        static void triSolve(vector<double> &dl, vector<double> &dm, vector<double> &du, vector<double> &rhs)
        {
            int n = (int)dm.size();
            for (int i = 1; i < n; ++i)
            {
                double w = dl[i - 1] / dm[i - 1];
                dm[i] -= w * du[i - 1];
                rhs[i] -= w * rhs[i - 1];
            }
            rhs[n - 1] /= dm[n - 1];
            for (int i = n - 2; i >= 0; --i)
                rhs[i] = (rhs[i] - du[i] * rhs[i + 1]) / dm[i];
        }

        void fit(const vector<double> &_s, const vector<double> &y)
        {
            int n = (int)_s.size();
            s = _s;
            a = y;
            b.assign(n, 0.0);
            c.assign(n, 0.0);
            d.assign(n, 0.0);
            if (n < 3)
            {
                if (n == 2)
                    b[0] = (a[1] - a[0]) / std::max(1e-30, s[1] - s[0]);
                return;
            }

            vector<double> h(n - 1);
            for (int i = 0; i < n - 1; ++i)
                h[i] = std::max(1e-30, s[i + 1] - s[i]);

            vector<double> dl(n - 2), dm(n - 2), du(n - 2), rhs(n - 2);
            for (int i = 1; i <= n - 2; ++i)
            {
                double hi_1 = h[i - 1], hi = h[i];
                dl[i - 1] = hi_1;
                dm[i - 1] = 2.0 * (hi_1 + hi);
                du[i - 1] = hi;
                rhs[i - 1] = 3.0 * ((a[i + 1] - a[i]) / hi - (a[i] - a[i - 1]) / hi_1);
            }
            if (n - 2 > 0)
                triSolve(dl, dm, du, rhs);
            for (int i = 1; i <= n - 2; ++i)
                c[i] = rhs[i - 1];
            c[0] = 0.0;
            c[n - 1] = 0.0;

            for (int i = 0; i < n - 1; ++i)
            {
                b[i] = (a[i + 1] - a[i]) / h[i] - (2.0 * c[i] + c[i + 1]) * h[i] / 3.0;
                d[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
            }
        }

        double eval(double si) const
        {
            int n = (int)s.size();
            if (n == 0)
                return 0.0;
            if (n == 1)
                return a[0];
            int lo = 0, hi = n - 1;
            if (si <= s.front())
                lo = 0;
            else if (si >= s.back())
                lo = n - 2;
            else
            {
                while (hi - lo > 1)
                {
                    int mid = (lo + hi) >> 1;
                    if (s[mid] <= si)
                        lo = mid;
                    else
                        hi = mid;
                }
            }
            double t = si - s[lo];
            return a[lo] + b[lo] * t + c[lo] * t * t + d[lo] * t * t * t;
        }

        void eval_with_deriv(double si, double &f, double &fp, double &fpp) const
        {
            int n = (int)s.size();
            if (n == 0)
            {
                f = fp = fpp = 0;
                return;
            }
            if (n == 1)
            {
                f = a[0];
                fp = fpp = 0;
                return;
            }

            int lo = 0, hi = n - 1;
            if (si <= s.front())
                lo = 0;
            else if (si >= s.back())
                lo = n - 2;
            else
            {
                while (hi - lo > 1)
                {
                    int mid = (lo + hi) >> 1;
                    if (s[mid] <= si)
                        lo = mid;
                    else
                        hi = mid;
                }
            }

            double t = si - s[lo];
            f = a[lo] + b[lo] * t + c[lo] * t * t + d[lo] * t * t * t;
            fp = b[lo] + 2.0 * c[lo] * t + 3.0 * d[lo] * t * t;
            fpp = 2.0 * c[lo] + 6.0 * d[lo] * t;
        }
    };

    inline vector<Vec2> splineUniformClosed_EXPORT_CONTEXT(
        const vector<Vec2> &ordered,
        int samples,
        int paddingK,
        bool close_loop, // 출력 중복 여부
        Spline1D &spx_out, Spline1D &spy_out,
        double &s0_out, double &L_out)
    {
        int N = (int)ordered.size();
        if (N < 3)
            return ordered;

        vector<Vec2> P;
        P.reserve(N + 2 * paddingK);
        for (int i = 0; i < paddingK; ++i)
            P.push_back(ordered[N - paddingK + i]);
        for (const auto &q : ordered)
            P.push_back(q);
        for (int i = 0; i < paddingK; ++i)
            P.push_back(ordered[i]);

        int M = (int)P.size();
        vector<double> s(M, 0.0), xs(M), ys(M);
        for (int i = 1; i < M; ++i)
        {
            double dx = P[i].x - P[i - 1].x, dy = P[i].y - P[i - 1].y;
            s[i] = s[i - 1] + std::sqrt(dx * dx + dy * dy);
        }
        for (int i = 0; i < M; ++i)
        {
            xs[i] = P[i].x;
            ys[i] = P[i].y;
        }

        Spline1D spx, spy;
        spx.fit(s, xs);
        spy.fit(s, ys);

        double s0 = s[paddingK], s1 = s[M - paddingK - 1];
        double L = std::max(1e-30, s1 - s0);

        vector<Vec2> out;
        out.reserve(samples + (close_loop ? 1 : 0));
        for (int k = 0; k < samples; k++)
        {
            double si = s0 + L * (double(k) / double(samples));
            out.push_back({spx.eval(si), spy.eval(si)});
        }
        if (close_loop)
            out.push_back(out.front());

        spx_out = std::move(spx);
        spy_out = std::move(spy);
        s0_out = s0;
        L_out = L;
        return out;
    }

    inline std::vector<int> orderIndicesByMST(const std::vector<geom::Vec2>& pts)
    {
        auto &C = cfg::get();
        int n = (int)pts.size();
        if (n <= 2) { std::vector<int> id(n); std::iota(id.begin(), id.end(), 0); return id; }

        int K = std::min(C.knn_k, n - 1);
        std::vector<std::vector<std::pair<int,double>>> adj(n);
        for (int i=0;i<n;i++){
            std::vector<std::pair<double,int>> cand; cand.reserve(n-1);
            for (int j=0;j<n;j++) if (i!=j){
                double d2=(pts[i].x-pts[j].x)*(pts[i].x-pts[j].x)+(pts[i].y-pts[j].y)*(pts[i].y-pts[j].y);
                cand.push_back({d2,j});
            }
            if ((int)cand.size()>K){
                std::nth_element(cand.begin(), cand.begin()+K, cand.end(),
                                [](auto&A,auto&B){return A.first<B.first;});
                cand.resize(K);
            }
            for (auto &c : cand){
                double w = std::sqrt(std::max(0.0, c.first));
                adj[i].push_back({c.second,w});
                adj[c.second].push_back({i,w});
            }
        }

        std::vector<double> key(n,1e300); std::vector<int> par(n,-1); std::vector<char> in(n,0);
        key[0]=0;
        for(int it=0;it<n;++it){
            int u=-1; double best=1e301;
            for(int i=0;i<n;i++) if(!in[i] && key[i]<best){best=key[i]; u=i;}
            if(u==-1) break; in[u]=1;
            for(auto [v,w]:adj[u]) if(!in[v] && w<key[v]){key[v]=w; par[v]=u;}
        }

        std::vector<std::vector<int>> tree(n);
        for (int v=0; v<n; v++) if (par[v]>=0){ tree[v].push_back(par[v]); tree[par[v]].push_back(v); }

        auto bfs = [&](int s){
            std::vector<double> d(n,1e300); std::vector<int> p(n,-1); std::queue<int> q;
            q.push(s); d[s]=0;
            while(!q.empty()){
                int u=q.front(); q.pop();
                for(int v:tree[u]) if(d[v]>1e299){
                    double w=std::hypot(pts[u].x-pts[v].x, pts[u].y-pts[v].y);
                    d[v]=d[u]+w; p[v]=u; q.push(v);
                }
            }
            int far=s; for(int i=0;i<n;i++) if(d[i]>d[far]) far=i;
            return std::tuple<int,std::vector<int>,std::vector<double>>(far,p,d);
        };
        auto [s1,p1,d1]=bfs(0);
        auto [s2,p2,d2]=bfs(s1);

        std::vector<int> path;
        for(int v=s2; v!=-1; v=p2[v]) path.push_back(v);
        std::vector<char> used(n,0);
        std::vector<int> out; out.reserve(n);
        for(int id: path){ out.push_back(id); used[id]=1; }
        for(int i=0;i<n;i++) if(!used[i]) out.push_back(i);
        return out;
    }
} // namespace centerline

// ---- forward declarations (Ray–Segment utils) ----
static double rayToRingDistance(const geom::Vec2 &P,
                                const geom::Vec2 &dir,
                                const std::vector<std::pair<geom::Vec2, geom::Vec2>> &ringEdges);

// ========================= MIN-CURV EXTENSION (raceline) =========================
namespace raceline_min_curv
{
    using geom::Vec2;

    inline int wrap(int i, int n)
    {
        i %= n;
        if (i < 0)
            i += n;
        return i;
    }

    struct DiffOps
    {
        int N;
        double h, inv2h, invh2;
        DiffOps(int N_, double h_) : N(N_), h(h_)
        {
            inv2h = 1.0 / (2.0 * h);
            invh2 = 1.0 / (h * h);
        }
        void D1(const std::vector<double> &a, std::vector<double> &out) const
        {
            out.resize(N);
            for (int i = 0; i < N; ++i)
            {
                int ip = wrap(i + 1, N), im = wrap(i - 1, N);
                out[i] = (a[ip] - a[im]) * inv2h;
            }
        }
        void D2(const std::vector<double> &a, std::vector<double> &out) const
        {
            out.resize(N);
            for (int i = 0; i < N; ++i)
            {
                int ip = wrap(i + 1, N), im = wrap(i - 1, N);
                out[i] = (a[ip] - 2.0 * a[i] + a[im]) * invh2;
            }
        }
        void D1T(const std::vector<double> &v, std::vector<double> &out) const
        {
            out.resize(N);
            for (int i = 0; i < N; ++i)
            {
                int im = wrap(i - 1, N), ip = wrap(i + 1, N);
                out[i] = (v[im] - v[ip]) * inv2h;
            }
        }
        void D2T(const std::vector<double> &v, std::vector<double> &out) const { D2(v, out); }
    };

    struct DiffOpsOpen
    {
        int N;
        double h, invh, inv2h, invh2;
        DiffOpsOpen(int N_, double h_) : N(N_), h(h_), invh(1.0 / h), inv2h(1.0 / (2 * h)), invh2(1.0 / (h * h)) {}
        void D1(const vector<double> &a, vector<double> &out) const
        {
            out.assign(N, 0.0);
            if (N == 0)
                return;
            if (N == 1)
            {
                out[0] = 0;
                return;
            }
            out[0] = (a[1] - a[0]) * invh;
            for (int i = 1; i <= N - 2; ++i)
                out[i] = (a[i + 1] - a[i - 1]) * inv2h;
            out[N - 1] = (a[N - 1] - a[N - 2]) * invh;
        }
        void D1T(const vector<double> &v, vector<double> &out) const
        {
            out.assign(N, 0.0);
            if (N == 0)
                return;
            if (N == 1)
            {
                out[0] = 0;
                return;
            }
            out[0] += (-invh) * v[0];
            out[1] += (+invh) * v[0];
            for (int i = 1; i <= N - 2; ++i)
            {
                out[i - 1] += (-inv2h) * v[i];
                out[i + 1] += (+inv2h) * v[i];
            }
            out[N - 2] += (-invh) * v[N - 1];
            out[N - 1] += (+invh) * v[N - 1];
        }
        void D2(const vector<double> &a, vector<double> &out) const
        {
            out.assign(N, 0.0);
            if (N <= 2)
                return;
            for (int i = 1; i <= N - 2; ++i)
                out[i] = (a[i + 1] - 2.0 * a[i] + a[i - 1]) * invh2;
        }
        void D2T(const vector<double> &v, vector<double> &out) const
        {
            out.assign(N, 0.0);
            if (N <= 2)
                return;
            for (int i = 1; i <= N - 2; ++i)
            {
                out[i - 1] += (+invh2) * v[i];
                out[i] += (-2.0 * invh2) * v[i];
                out[i + 1] += (+invh2) * v[i];
            }
        }
    };

    static void normals_from_points_generic(const std::vector<Vec2> &P, bool closed, std::vector<Vec2> &n)
    {
        int N = (int)P.size();
        n.assign(N, {0, 0});
        if (N == 0)
            return;
        auto t_of = [&](int i) -> Vec2
        {
            if (N == 1)
                return {1, 0};
            if (closed)
            {
                int ip = (i + 1) % N, im = (i - 1 + N) % N;
                return {(P[ip].x - P[im].x) * 0.5, (P[ip].y - P[im].y) * 0.5};
            }
            else
            {
                if (i == 0)
                    return {P[1].x - P[0].x, P[1].y - P[0].y};
                if (i == N - 1)
                    return {P[N - 1].x - P[N - 2].x, P[N - 1].y - P[N - 2].y};
                return {(P[i + 1].x - P[i - 1].x) * 0.5, (P[i + 1].y - P[i - 1].y) * 0.5};
            }
        };
        for (int i = 0; i < N; ++i)
        {
            Vec2 t = t_of(i);
            if (geom::norm(t) < 1e-15)
                t = {1, 0};
            Vec2 nv{-t.y, t.x};
            n[i] = geom::normalize(nv, 1e-15);
        }
    }

    static void heading_curv_from_points_generic(const std::vector<Vec2> &P, double h, bool closed,
                                                 std::vector<double> &heading, std::vector<double> &kappa)
    {
        int N = (int)P.size();
        heading.assign(N, 0.0);
        kappa.assign(N, 0.0);
        if (N == 0)
            return;
        auto deriv = [&](int i, double &xp, double &yp, double &xpp, double &ypp)
        {
            if (N == 1)
            {
                xp = 1;
                yp = 0;
                xpp = ypp = 0;
                return;
            }
            if (closed)
            {
                int ip = (i + 1) % N, im = (i - 1 + N) % N;
                xp = (P[ip].x - P[im].x) / (2 * h);
                yp = (P[ip].y - P[im].y) / (2 * h);
                xpp = (P[ip].x - 2.0 * P[i].x + P[im].x) / (h * h);
                ypp = (P[ip].y - 2.0 * P[i].y + P[im].y) / (h * h);
            }
            else
            {
                if (i == 0)
                {
                    xp = (P[1].x - P[0].x) / h;
                    yp = (P[1].y - P[0].y) / h;
                    if (N >= 3)
                    {
                        xpp = (P[2].x - 2.0 * P[1].x + P[0].x) / (h * h);
                        ypp = (P[2].y - 2.0 * P[1].y + P[0].y) / (h * h);
                    }
                    else
                        xpp = ypp = 0;
                }
                else if (i == N - 1)
                {
                    xp = (P[N - 1].x - P[N - 2].x) / h;
                    yp = (P[N - 1].y - P[N - 2].y) / h;
                    if (N >= 3)
                    {
                        xpp = (P[N - 1].x - 2.0 * P[N - 2].x + P[N - 3].x) / (h * h);
                        ypp = (P[N - 1].y - 2.0 * P[N - 2].y + P[N - 3].y) / (h * h);
                    }
                    else
                        xpp = ypp = 0;
                }
                else
                {
                    xp = (P[i + 1].x - P[i - 1].x) / (2 * h);
                    yp = (P[i + 1].y - P[i - 1].y) / (2 * h);
                    xpp = (P[i + 1].x - 2.0 * P[i].x + P[i - 1].x) / (h * h);
                    ypp = (P[i + 1].y - 2.0 * P[i].y + P[i - 1].y) / (h * h);
                }
            }
        };
        for (int i = 0; i < N; ++i)
        {
            double xp, yp, xpp, ypp;
            deriv(i, xp, yp, xpp, ypp);
            heading[i] = std::atan2(yp, xp);
            double denom = std::pow(std::max(1e-12, xp * xp + yp * yp), 1.5);
            kappa[i] = (xp * ypp - yp * xpp) / denom;
        }
    }

    struct LinGeom
    {
        std::vector<double> A1, A2, N0, W;
    };

    static LinGeom precompute_lin_geom_generic(const std::vector<Vec2> &Pbase,
                                               const std::vector<Vec2> &n,
                                               double h, bool closed)
    {
        int N = (int)Pbase.size();
        std::vector<double> xp(N), yp(N), xpp(N), ypp(N);
        auto deriv = [&](int i, double &_xp, double &_yp, double &_xpp, double &_ypp)
        {
            if (N == 1)
            {
                _xp = 1;
                _yp = 0;
                _xpp = _ypp = 0;
                return;
            }
            if (closed)
            {
                int ip = (i + 1) % N, im = (i - 1 + N) % N;
                _xp = (Pbase[ip].x - Pbase[im].x) / (2 * h);
                _yp = (Pbase[ip].y - Pbase[im].y) / (2 * h);
                _xpp = (Pbase[ip].x - 2.0 * Pbase[i].x + Pbase[im].x) / (h * h);
                _ypp = (Pbase[ip].y - 2.0 * Pbase[i].y + Pbase[im].y) / (h * h);
            }
            else
            {
                if (i == 0)
                {
                    _xp = (Pbase[1].x - Pbase[0].x) / h;
                    _yp = (Pbase[1].y - Pbase[0].y) / h;
                    if (N >= 3)
                    {
                        _xpp = (Pbase[2].x - 2.0 * Pbase[1].x + Pbase[0].x) / (h * h);
                        _ypp = (Pbase[2].y - 2.0 * Pbase[1].y + Pbase[0].y) / (h * h);
                    }
                    else
                        _xpp = _ypp = 0;
                }
                else if (i == N - 1)
                {
                    _xp = (Pbase[N - 1].x - Pbase[N - 2].x) / h;
                    _yp = (Pbase[N - 1].y - Pbase[N - 2].y) / h;
                    if (N >= 3)
                    {
                        _xpp = (Pbase[N - 1].x - 2.0 * Pbase[N - 2].x + Pbase[N - 3].x) / (h * h);
                        _ypp = (Pbase[N - 1].y - 2.0 * Pbase[N - 2].y + Pbase[N - 3].y) / (h * h);
                    }
                    else
                        _xpp = _ypp = 0;
                }
                else
                {
                    _xp = (Pbase[i + 1].x - Pbase[i - 1].x) / (2 * h);
                    _yp = (Pbase[i + 1].y - Pbase[i - 1].y) / (2 * h);
                    _xpp = (Pbase[i + 1].x - 2.0 * Pbase[i].x + Pbase[i - 1].x) / (h * h);
                    _ypp = (Pbase[i + 1].y - 2.0 * Pbase[i].y + Pbase[i - 1].y) / (h * h);
                }
            }
        };
        for (int i = 0; i < N; ++i)
            deriv(i, xp[i], yp[i], xpp[i], ypp[i]);

        LinGeom G;
        G.A1.resize(N);
        G.A2.resize(N);
        G.N0.resize(N);
        G.W.resize(N);
        for (int i = 0; i < N; ++i)
        {
            G.A1[i] = n[i].x * ypp[i] - n[i].y * xpp[i];
            G.A2[i] = xp[i] * n[i].y - yp[i] * n[i].x;
            G.N0[i] = xp[i] * ypp[i] - yp[i] * xpp[i];
            double denom = std::pow(std::max(1e-12, xp[i] * xp[i] + yp[i] * yp[i]), 1.5);
            G.W[i] = 1.0 / denom;
        }
        return G;
    }

    struct CostGrad
    {
        double J;
        std::vector<double> grad;
    };

    static CostGrad eval_cost_grad_frozen(const std::vector<double> &A1,
                                          const std::vector<double> &A2,
                                          const std::vector<double> &N0,
                                          const std::vector<double> &W,
                                          double h, double lambda_smooth,
                                          const std::vector<double> &alpha,
                                          bool closed)
    {
        int N = (int)alpha.size();
        std::vector<double> a1, a2;

        if (closed)
        {
            DiffOps D(N, h);
            D.D1(alpha, a1);
            D.D2(alpha, a2);
        }
        else
        {
            DiffOpsOpen D(N, h);
            D.D1(alpha, a1);
            D.D2(alpha, a2);
        }

        std::vector<double> z(N);
        for (int i = 0; i < N; ++i)
            z[i] = W[i] * (N0[i] + A1[i] * a1[i] + A2[i] * a2[i]);

        double J = 0.0;
        for (double v : z)
            J += v * v;
        double Jsm = 0.0;
        for (double v : a1)
            Jsm += v * v;
        J += lambda_smooth * Jsm;

        std::vector<double> Wz(N), q1(N), q2(N), g1, g2, gsm, D1a;
        for (int i = 0; i < N; ++i)
        {
            Wz[i] = W[i] * z[i];
            q1[i] = A1[i] * Wz[i];
            q2[i] = A2[i] * Wz[i];
        }

        if (closed)
        {
            DiffOps D(N, h);
            D.D1T(q1, g1);
            D.D2T(q2, g2);
            D.D1(alpha, D1a);
            D.D1T(D1a, gsm);
        }
        else
        {
            DiffOpsOpen D(N, h);
            D.D1T(q1, g1);
            D.D2T(q2, g2);
            D.D1(alpha, D1a);
            D.D1T(D1a, gsm);
        }

        std::vector<double> grad(N);
        for (int i = 0; i < N; ++i)
            grad[i] = 2.0 * (g1[i] + g2[i]) + 2.0 * lambda_smooth * gsm[i];
        return {J, std::move(grad)};
    }

    struct Result
    {
        std::vector<Vec2> raceline;
        std::vector<double> heading;
        std::vector<double> curvature;
        std::vector<double> alpha_total;
        std::vector<double> alpha_last;
    };

    static double minDistanceToSegments(const Vec2 &P,
                                        const std::vector<std::pair<Vec2, Vec2>> &E)
    {
        double best = std::numeric_limits<double>::infinity();
        for (const auto &e : E)
        {
            Vec2 a = e.first, b = e.second;
            Vec2 ab{b.x - a.x, b.y - a.y}, ap{P.x - a.x, P.y - a.y};
            double denom = std::max(1e-30, ab.x * ab.x + ab.y * ab.y);
            double t = std::clamp((ab.x * ap.x + ab.y * ap.y) / denom, 0.0, 1.0);
            Vec2 Q{a.x + ab.x * t, a.y + ab.y * t};
            best = std::min(best, std::hypot(P.x - Q.x, P.y - Q.y));
        }
        return best;
    }

    static Result compute_min_curvature_raceline(const std::vector<Vec2> &center,
                                                 const std::vector<std::pair<Vec2, Vec2>> &innerE,
                                                 const std::vector<std::pair<Vec2, Vec2>> &outerE,
                                                 double veh_width,
                                                 double L,
                                                 bool closed)
    {
        auto &C = cfg::get();
        int N = (int)center.size();
        if (N == 0)
            return {};

        double h = L / double(N);

        std::vector<Vec2> Pbase = center;
        std::vector<Vec2> n;
        normals_from_points_generic(Pbase, closed, n);

        // --- helper: 안전한 레이 거리(없으면 최근접 선분 거리 fallback) ---
        auto safe_ray = [&](const Vec2 &P0, const Vec2 &dir, const std::vector<std::pair<Vec2, Vec2>> &E)
        {
            double t = ::rayToRingDistance(P0, dir, E);
            if (!std::isfinite(t))
                t = minDistanceToSegments(P0, E);
            if (!std::isfinite(t))
                t = 0.0;
            return std::max(0.0, t);
        };

        std::vector<double> lo, hi;
        lo.assign(N, 0.0);
        hi.assign(N, 0.0);
        for (int i = 0; i < N; ++i)
        {
            Vec2 nv = n[i], P0 = Pbase[i], nneg{-nv.x, -nv.y};

            double dpos_in = safe_ray(P0, nv, innerE);
            double dpos_out = safe_ray(P0, nv, outerE);
            double dneg_in = safe_ray(P0, nneg, innerE);
            double dneg_out = safe_ray(P0, nneg, outerE);

            double dpos = std::min(dpos_in, dpos_out);
            double dneg = std::min(dneg_in, dneg_out);
            double guard = veh_width * 0.5 + C.safety_margin_m;

            hi[i] = std::max(0.0, dpos - guard);
            lo[i] = -std::max(0.0, dneg - guard);
            if (!std::isfinite(hi[i]))
                hi[i] = 0.0;
            if (!std::isfinite(lo[i]))
                lo[i] = 0.0;
        }

        if (true)
        {
            double hi_avg = 0, lo_avg = 0, hi_max = 0, lo_max = 0;
            for (int i = 0; i < N; ++i)
            {
                hi_avg += hi[i];
                lo_avg += -lo[i];
                hi_max = std::max(hi_max, hi[i]);
                lo_max = std::max(lo_max, -lo[i]);
            }
            hi_avg /= std::max(1, N);
            lo_avg /= std::max(1, N);
            cerr << "[corridor] mean+ = " << hi_avg << "  mean- = " << lo_avg
                 << "  max+ = " << hi_max << "  max- = " << lo_max << "\n";
        }

        std::vector<double> alpha(N, 0.0), alpha_accum(N, 0.0), alpha_last_stage(N, 0.0);

        for (int outer = 0; outer < C.max_outer_iters; ++outer)
        {
            auto G = precompute_lin_geom_generic(Pbase, n, h, closed);

            double step = C.step_init;
            auto cg = eval_cost_grad_frozen(G.A1, G.A2, G.N0, G.W, h, C.lambda_smooth, alpha, closed);
            double J_prev = cg.J;
            if (true)
                cerr << "[GN " << outer << "]  J0=" << J_prev << "  step=" << step << "  lambda=" << C.lambda_smooth << "\n";

            for (int it = 0; it < C.max_inner_iters; ++it)
            {
                bool accepted = false;
                int bt = 0;
                while (bt < 20)
                {
                    std::vector<double> a_new(N);
                    for (int i = 0; i < N; ++i)
                    {
                        double ai = alpha[i] - step * cg.grad[i];
                        a_new[i] = std::min(hi[i], std::max(lo[i], ai));
                    }
                    auto cg_new = eval_cost_grad_frozen(G.A1, G.A2, G.N0, G.W, h, C.lambda_smooth, a_new, closed);

                    double dec = 0.0;
                    for (int i = 0; i < N; ++i)
                        dec += cg.grad[i] * (a_new[i] - alpha[i]);
                    if (cg_new.J <= cg.J + C.armijo_c * dec)
                    {
                        alpha.swap(a_new);
                        cg = std::move(cg_new);
                        accepted = true;
                        break;
                    }
                    step *= 0.5;
                    bt++;
                    if (step < C.step_min)
                        break;
                }
                if (!accepted)
                    break;
                if (std::fabs(J_prev - cg.J) < 1e-10)
                    break;
                J_prev = cg.J;
            }
            alpha_last_stage = alpha;

            for (int i = 0; i < N; ++i)
            {
                Pbase[i].x += n[i].x * alpha[i];
                Pbase[i].y += n[i].y * alpha[i];
                alpha_accum[i] += alpha[i];
            }
            normals_from_points_generic(Pbase, closed, n);

            // 새 코리도 재평가
            for (int i = 0; i < N; ++i)
            {
                Vec2 nv = n[i], P0 = Pbase[i], nneg{-nv.x, -nv.y};
                double dpos = std::min(safe_ray(P0, nv, innerE), safe_ray(P0, nv, outerE));
                double dneg = std::min(safe_ray(P0, nneg, innerE), safe_ray(P0, nneg, outerE));
                double guard = veh_width * 0.5 + C.safety_margin_m;
                hi[i] = std::max(0.0, dpos - guard);
                lo[i] = -std::max(0.0, dneg - guard);
                if (!std::isfinite(hi[i]))
                    hi[i] = 0.0;
                if (!std::isfinite(lo[i]))
                    lo[i] = 0.0;
            }
            std::fill(alpha.begin(), alpha.end(), 0.0);
        }

        std::vector<double> heading, kappa;
        heading_curv_from_points_generic(Pbase, h, closed, heading, kappa);

        if (true)
        {
            double ksum = 0;
            for (double v : kappa)
                ksum += v * v;
            cerr << "[done] ∑κ^2 = " << ksum << "  (with λ=" << cfg::get().lambda_smooth << ")\n";
        }

        return {std::move(Pbase), std::move(heading), std::move(kappa), std::move(alpha_accum), std::move(alpha_last_stage)};
    }

} // namespace raceline_min_curv
// ======================= END: MIN-CURV EXTENSION (raceline) =======================

//=============================== IO Utils ==================================
namespace io
{
    using geom::Vec2;

    inline vector<Vec2> loadCSV_XY(const string &path)
    {
        vector<Vec2> pts;
        std::ifstream fin(path);
        if (!fin)
        {
            cerr << "[ERR] cannot open: " << path << "\n";
            return pts;
        }
        string line;
        while (std::getline(fin, line))
        {
            if (line.empty())
                continue;
            for (char &ch : line)
                if (ch == ';' || ch == '\t')
                    ch = ' ';
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream iss(line);
            double x, y;
            if (iss >> x >> y)
                pts.push_back({x, y});
        }
        return pts;
    }
    inline bool saveCSV_pointsXY(const string &path, const vector<Vec2> &pts)
    {
        std::ofstream fo(path);
        if (!fo)
        {
            cerr << "[ERR] write failed: " << path << "\n";
            return false;
        }
        fo.setf(std::ios::fixed);
        fo.precision(9);
        for (auto &p : pts)
            fo << p.x << "," << p.y << "\n";
        return true;
    }
    inline bool saveCSV_pointsLabeled(const string &path, const vector<Vec2> &pts, const vector<int> &label)
    {
        std::ofstream fo(path);
        if (!fo)
        {
            cerr << "[ERR] write failed: " << path << "\n";
            return false;
        }
        fo << "id,x,y,label\n";
        fo.setf(std::ios::fixed);
        fo.precision(9);
        for (size_t i = 0; i < pts.size(); ++i)
            fo << i << "," << pts[i].x << "," << pts[i].y << "," << label[i] << "\n";
        return true;
    }
    inline bool saveCSV_edgesIdx(const string &path, const vector<pair<int, int>> &E)
    {
        std::ofstream fo(path);
        if (!fo)
        {
            cerr << "[ERR] write failed: " << path << "\n";
            return false;
        }
        for (auto &e : E)
            fo << e.first << "," << e.second << "\n";
        return true;
    }
    inline bool saveCSV_trisIdx(const string &path, const vector<delaunay::Tri> &T)
    {
        std::ofstream fo(path);
        if (!fo)
        {
            cerr << "[ERR] write failed: " << path << "\n";
            return false;
        }
        for (auto &t : T)
            fo << t.a << "," << t.b << "," << t.c << "\n";
        return true;
    }
    inline string dropExt(const string &s)
    {
        size_t p = s.find_last_of('.');
        if (p == string::npos)
            return s;
        return s.substr(0, p);
    }
} // namespace io

//============================ Ray–Segment Utils ============================
// 레이 A + t*d 와 세그먼트 S0 + u*(S1-S0) 교차 (t>=0, u∈[0,1])
static bool rayIntersectSegment(const geom::Vec2 &A, const geom::Vec2 &d,
                                const geom::Vec2 &S0, const geom::Vec2 &S1,
                                double &t_out, double eps = 1e-15)
{
    double vx = S1.x - S0.x, vy = S1.y - S0.y;
    double den = d.x * (-vy) + d.y * (vx);
    if (std::fabs(den) < eps)
        return false;

    double ax = S0.x - A.x, ay = S0.y - A.y;
    double inv = 1.0 / den;
    double t = (ax * (-vy) + ay * (vx)) * inv;
    double u = (d.x * ay - d.y * ax) * inv;

    if (t >= 0.0 && u >= -1e-12 && u <= 1.0 + 1e-12)
    {
        t_out = t;
        return true;
    }
    return false;
}

static double rayToRingDistance(const geom::Vec2 &P, const geom::Vec2 &dir,
                                const vector<pair<geom::Vec2, geom::Vec2>> &ringEdges)
{
    double best = std::numeric_limits<double>::infinity();
    for (const auto &e : ringEdges)
    {
        double t;
        if (rayIntersectSegment(P, dir, e.first, e.second, t))
            if (t > 0.0 && t < best)
                best = t;
    }
    return best;
}

// 최근접 선분 거리 (fallback)
static double minDistanceToSegments_global(const geom::Vec2 &P, const std::vector<std::pair<geom::Vec2, geom::Vec2>> &E)
{
    double best = std::numeric_limits<double>::infinity();
    for (const auto &e : E)
    {
        geom::Vec2 a = e.first, b = e.second;
        geom::Vec2 ab{b.x - a.x, b.y - a.y}, ap{P.x - a.x, P.y - a.y};
        double denom = std::max(1e-30, ab.x * ab.x + ab.y * ab.y);
        double t = std::clamp((ab.x * ap.x + ab.y * ap.y) / denom, 0.0, 1.0);
        geom::Vec2 Q{a.x + ab.x * t, a.y + ab.y * t};
        best = std::min(best, std::hypot(P.x - Q.x, P.y - Q.y));
    }
    return best;
}

// 한 점 P와 법선 n에 대해 inner/outer까지의 거리(d_inner, d_outer)
// 레이가 실패하면 최근접 선분 거리로 fallback
static void distancesToRings(const geom::Vec2 &P, const geom::Vec2 &n,
                             const vector<pair<geom::Vec2, geom::Vec2>> &innerE,
                             const vector<pair<geom::Vec2, geom::Vec2>> &outerE,
                             double &d_inner, double &d_outer)
{
    geom::Vec2 npos = n;
    geom::Vec2 nneg = geom::Vec2{-n.x, -n.y};

    double di1 = rayToRingDistance(P, npos, innerE);
    double di2 = rayToRingDistance(P, nneg, innerE);
    if (!std::isfinite(di1) && !std::isfinite(di2))
        d_inner = minDistanceToSegments_global(P, innerE);
    else
        d_inner = std::min(di1, di2);

    double do1 = rayToRingDistance(P, npos, outerE);
    double do2 = rayToRingDistance(P, nneg, outerE);
    if (!std::isfinite(do1) && !std::isfinite(do2))
        d_outer = minDistanceToSegments_global(P, outerE);
    else
        d_outer = std::min(do1, do2);
    if (!std::isfinite(d_inner))
        d_inner = 0.0;
    if (!std::isfinite(d_outer))
        d_outer = 0.0;
}

static inline geom::Vec2 dir_from_heading_rad(double rad)
{
    return {std::cos(rad), std::sin(rad)};
}

#include <chrono>

//============================= Timing Utils ================================
namespace timing
{
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    struct Scoped
    {
        const char *name;
        Clock::time_point t0;
        Scoped(const char *n) : name(n), t0(Clock::now()) {}
        ~Scoped()
        {
            auto ms = std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
            std::cerr.setf(std::ios::fixed);
            std::cerr << "[TIME] " << name << " = " << std::setprecision(3) << ms << " ms\n";
        }
    };
    struct ScopedAcc
    {
        const char *name;
        double *acc;
        Clock::time_point t0;
        ScopedAcc(const char *n, double *a) : name(n), acc(a), t0(Clock::now()) {}
        ~ScopedAcc()
        {
            auto ms = std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
            if (acc)
                *acc += ms;
            std::cerr.setf(std::ios::fixed);
            std::cerr << "[TIME] " << name << " = " << std::setprecision(3) << ms << " ms\n";
        }
    };
}

//==================================== MAIN =================================
int main(int argc, char **argv)
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // 전체 실행 시간 시작
    auto _t_all = timing::Clock::now();
    double T_load = 0, T_order = 0, T_dt = 0, T_mids = 0, T_orderMST = 0,
           T_spline = 0, T_saveCenter = 0, T_geom = 0, T_race = 0, T_saveRace = 0;

    if (argc < 4)
    {
        cerr << "Usage: " << argv[0] << " inner.csv outer.csv centerline.csv\n";
        return 1;
    }
    std::string innerPath = argv[1], outerPath = argv[2], outPath = argv[3];
    const std::string base = io::dropExt(outPath);

    auto &C = cfg::get();

    // [1] 입력 로드
    std::vector<geom::Vec2> inner, outer;
    {
        timing::ScopedAcc _t("1) 입력 로드", &T_load);
        inner = io::loadCSV_XY(innerPath);
        outer = io::loadCSV_XY(outerPath);
    }
    if (inner.size() < 2 || outer.size() < 2)
    {
        cerr << "[ERR] need >= 2 points per ring (open에서도 최소 2)\n";
        return 2;
    }

    const bool closed_mode = C.is_closed_track;

    // [DBG] 현재/초기 앵커 상태 출력
    {
        const geom::Vec2 start_anchor{C.start_anchor_x, C.start_anchor_y};
        const geom::Vec2 curr_pos{C.current_pos_x, C.current_pos_y};
        const geom::Vec2 curr_dir = geom::normalize(dir_from_heading_rad(C.current_heading_rad), 1e-12);
        std::cerr.setf(std::ios::fixed);
        std::cerr << std::setprecision(9)
                  << "[DBG] start_anchor=(" << start_anchor.x << "," << start_anchor.y << ") "
                  << "curr_pos=(" << curr_pos.x << "," << curr_pos.y << ") "
                  << "curr_heading_rad=" << C.current_heading_rad
                  << " curr_dir=(" << curr_dir.x << "," << curr_dir.y << ")\n";
    }

    // [2] Delaunay(Bowyer–Watson) — DT only
    std::vector<geom::Vec2> all_pts;
    std::vector<int> labels; // 0=inner, 1=outer
    std::vector<delaunay::Tri> tris;
    {
        timing::ScopedAcc _t("2) Delaunay 삼각분할(BW)", &T_dt);
        all_pts = inner;
        all_pts.insert(all_pts.end(), outer.begin(), outer.end());
        labels.assign(all_pts.size(), 0);
        for (size_t i = 0; i < all_pts.size(); ++i)
            labels[i] = (i < inner.size() ? 0 : 1);

        tris = delaunay::bowyerWatson(all_pts);
    }

    // 포인트+라벨 / 삼각형 / 모든 엣지 저장
    io::saveCSV_pointsLabeled(base + "_all_points.csv", all_pts, labels);
    io::saveCSV_trisIdx(base + "_tri_raw_idx.csv", tris);
    {
        std::unordered_map<delaunay::EdgeKey, std::vector<delaunay::EdgeRef>, delaunay::EdgeKeyHash> M;
        delaunay::buildEdgeMap(tris, M);
        std::vector<std::pair<int,int>> edges_all;
        edges_all.reserve(M.size());
        for (const auto& kv : M)
            edges_all.push_back({kv.first.u, kv.first.v});
        io::saveCSV_edgesIdx(base + "_edges_all_idx.csv", edges_all);
    }

    // [3] 라벨 다른 경계엣지 추출 + 길이 필터
    std::vector<geom::Vec2> mids;
    std::vector<int> keep_edge_idx; // 길이필터 통과한 경계엣지의 binfo 인덱스
    std::vector<centerline::BoundaryEdgeInfo> binfo; // 재사용/저장용
    {
        timing::ScopedAcc _t("3) 경계엣지 중점+길이필터", &T_mids);
        binfo = centerline::labelBoundaryEdges_with_len(all_pts, tris, labels);
        if (binfo.empty())
        {
            cerr << "[ERR] no label-different boundary edges\n";
            return 4;
        }

        // 라벨 다른 엣지 저장
        std::vector<std::pair<int,int>> edges_mixed;
        edges_mixed.reserve(binfo.size());
        for (auto &e : binfo) edges_mixed.push_back({e.u, e.v});
        io::saveCSV_edgesIdx(base + "_edges_labeldiff_idx.csv", edges_mixed);

        // 길이 중앙값 기반 컷오프
        std::vector<double> lens; lens.reserve(binfo.size());
        for (auto &e : binfo) lens.push_back(e.len);
        std::sort(lens.begin(), lens.end());
        auto quant = [&](double p)->double{
            if (lens.empty()) return 0.0;
            double idx = p * (lens.size()-1);
            size_t i = (size_t)std::floor(idx);
            size_t j = std::min(i+1, lens.size()-1);
            double t = idx - i;
            return (1.0-t)*lens[i] + t*lens[j];
        };
        double Lmed = quant(0.5);
        bool apply = cfg::get().enable_boundary_len_filter;
        double cutoff = std::min(cfg::get().boundary_edge_abs_max,
                                 cfg::get().boundary_edge_len_scale * std::max(1e-12, Lmed));

        // 필터 통과 중점/인덱스
        mids.reserve(binfo.size());
        keep_edge_idx.reserve(binfo.size());
        for (int i=0;i<(int)binfo.size();++i){
            const auto &e = binfo[i];
            if (!apply || e.len <= cutoff){
                mids.push_back(e.mid);
                keep_edge_idx.push_back(i);
            }
        }
        if (mids.size() < 2)
        {
            cerr << "[ERR] not enough midpoints after length filter (" << mids.size() << "), adjust thresholds.\n";
            io::saveCSV_pointsXY(base + "_mids_raw.csv", mids);
            return 4;
        }

        // 길이 필터 통과한 엣지 저장
        std::vector<std::pair<int,int>> edges_kept;
        edges_kept.reserve(keep_edge_idx.size());
        for (int idx : keep_edge_idx){
            const auto &e = binfo[idx];
            edges_kept.push_back({e.u, e.v});
        }
        io::saveCSV_edgesIdx(base + "_edges_labeldiff_kept_idx.csv", edges_kept);
    }

    // 샘플 수 동적 조정
    if (C.use_dynamic_samples)
    {
        int total_mids = (int)mids.size();
        int dyn = (int)std::llround(C.sample_factor_n * std::max(0, total_mids));
        dyn = std::max(dyn, C.samples_min);
        if (C.samples_max > 0) dyn = std::min(dyn, C.samples_max);
        dyn = std::max(dyn, 4);
        C.samples = dyn;
    }

    // [4] 중점 순서화(MST) + 방향/시작점 정합
    std::vector<geom::Vec2> ordered;
    std::vector<int> mids_order_idx;
    {
        timing::ScopedAcc _t("4) 중점 순서화(MST)+방향/시작점 정합", &T_orderMST);

        ordered = centerline::orderByMST(mids);
        mids_order_idx = centerline::orderIndicesByMST(mids);

        // ---- 유틸 ----
        auto reverse_both = [&](std::vector<geom::Vec2>& A, std::vector<int>& B){
            std::reverse(A.begin(), A.end());
            std::reverse(B.begin(), B.end());
        };
        auto dist2 = [](const geom::Vec2& p, const geom::Vec2& q){
            double dx=p.x-q.x, dy=p.y-q.y; return dx*dx+dy*dy;
        };
        auto nearest_idx = [&](const std::vector<geom::Vec2>& S, const geom::Vec2& target)->size_t{
            size_t k=0; double best=1e300;
            for(size_t i=0;i<S.size();++i){ double d=dist2(S[i], target); if(d<best){best=d; k=i;} }
            return k;
        };
        auto vnorm = [](const geom::Vec2& v)->geom::Vec2{
            return geom::normalize(v, 1e-12);
        };
        auto local_dir_open = [&](const std::vector<geom::Vec2>& S, size_t i)->geom::Vec2{
            if (S.size()<2) return {1,0};
            if (i+1 < S.size()) return vnorm(geom::Vec2{S[i+1].x - S[i].x, S[i+1].y - S[i].y});
            return vnorm(geom::Vec2{S[i].x - S[i-1].x, S[i].y - S[i-1].y});
        };
        auto local_dir_closed = [&](const std::vector<geom::Vec2>& S, size_t i)->geom::Vec2{
            if (S.size()<2) return {1,0};
            size_t j = (i+1) % S.size();
            return vnorm(geom::Vec2{S[j].x - S[i].x, S[j].y - S[i].y});
        };

        if (!closed_mode) {
            // ===== OPEN =====
            // 1) 방향: 현재 차량 위치에 가장 가까운 중점 i*의 국소 진행방향 vs 현재 헤딩
            geom::Vec2 curr_pos{C.current_pos_x, C.current_pos_y};
            geom::Vec2 curr_dir = vnorm(dir_from_heading_rad(C.current_heading_rad));

            size_t i_near_cur = nearest_idx(ordered, curr_pos);
            geom::Vec2 vloc = local_dir_open(ordered, i_near_cur);
            double dotv = vloc.x*curr_dir.x + vloc.y*curr_dir.y;
            if (dotv < 0.0)
                reverse_both(ordered, mids_order_idx);

            // 2) 시작점: 초기 위치(start_anchor)에 가장 가까운 중점이 index 0이 되도록 회전
            geom::Vec2 start_anchor{C.start_anchor_x, C.start_anchor_y};
            size_t k = nearest_idx(ordered, start_anchor);
            std::rotate(ordered.begin(), ordered.begin()+k, ordered.end());
            std::rotate(mids_order_idx.begin(), mids_order_idx.begin()+k, mids_order_idx.end());
        } else {
            // ===== CLOSED =====
            // 1) 방향: 초기 위치에 가장 가까운 중점 i*의 wrap-다음점 방향 vs 초기 헤딩
            geom::Vec2 start_anchor{C.start_anchor_x, C.start_anchor_y};
            geom::Vec2 init_dir = vnorm(dir_from_heading_rad(C.start_heading_rad));

            size_t i_near_anchor = nearest_idx(ordered, start_anchor);
            geom::Vec2 vloc = local_dir_closed(ordered, i_near_anchor);
            double dotv = vloc.x*init_dir.x + vloc.y*init_dir.y;
            if (dotv < 0.0)
                reverse_both(ordered, mids_order_idx);

            // 2) 시작점: start_anchor에 가장 가까운 중점이 index 0이 되도록 회전
            i_near_anchor = nearest_idx(ordered, start_anchor); // 뒤집혔을 수 있으니 재계산
            std::rotate(ordered.begin(), ordered.begin()+i_near_anchor, ordered.end());
            std::rotate(mids_order_idx.begin(), mids_order_idx.begin()+i_near_anchor, mids_order_idx.end());
        }

        io::saveCSV_pointsXY(base + "_mids_raw.csv", mids);
        io::saveCSV_pointsXY(base + "_mids_ordered.csv", ordered);
    }

    // [5] 중점 순서를 이용해 링(콘) 재정렬 + 방향(중점 정렬 방향과 일치)
    std::vector<geom::Vec2> inner_from_mids, outer_from_mids;
    {
        std::vector<char> seen_inner(all_pts.size(), 0);
        std::vector<char> seen_outer(all_pts.size(), 0);

        auto get_inner_outer = [&](int u, int v, int &iv, int &ov){
            // label: 0=inner, 1=outer
            if (labels[u] == 0 && labels[v] == 1){ iv = u; ov = v; }
            else if (labels[u] == 1 && labels[v] == 0){ iv = v; ov = u; }
            else { iv = ov = -1; }
        };

        for (int k=0; k<(int)mids_order_idx.size(); ++k){
            int mid_local_idx = mids_order_idx[k];    // mids[] 기준 인덱스
            int eidx = keep_edge_idx[mid_local_idx];  // binfo 인덱스
            int u = binfo[eidx].u, v = binfo[eidx].v;

            int iv=-1, ov=-1;
            get_inner_outer(u, v, iv, ov);
            if (iv>=0 && !seen_inner[iv]){ inner_from_mids.push_back(all_pts[iv]); seen_inner[iv]=1; }
            if (ov>=0 && !seen_outer[ov]){ outer_from_mids.push_back(all_pts[ov]); seen_outer[ov]=1; }
        }

        // ====== 방향 정리 (중점 방향에 맞춤) ======
        auto vnorm = [&](const geom::Vec2& a)->geom::Vec2{
            return geom::normalize(a, 1e-12);
        };
        auto first_dir = [&](const std::vector<geom::Vec2>& S)->geom::Vec2{
            if (S.size()<2) return geom::Vec2{1,0};
            return vnorm(geom::Vec2{S[1].x - S[0].x, S[1].y - S[0].y});
        };
        // 중점 진행 방향(기준)
        geom::Vec2 v_ref = first_dir(ordered);

        auto align_to_mids_dir = [&](std::vector<geom::Vec2>& seq){
            if (seq.size()<2) return;
            geom::Vec2 v_seq = first_dir(seq);
            double d = v_ref.x * v_seq.x + v_ref.y * v_seq.y;
            if (d < 0.0) std::reverse(seq.begin(), seq.end());
        };

        // inner/outer 모두 중점 방향에 정렬
        align_to_mids_dir(inner_from_mids);
        align_to_mids_dir(outer_from_mids);

        // 저장(디버깅/재사용)
        io::saveCSV_pointsXY(base + "_inner_from_mids.csv", inner_from_mids);
        io::saveCSV_pointsXY(base + "_outer_from_mids.csv", outer_from_mids);
    }

    // [6] 스플라인 + 균일 재샘플 (센터라인)
    centerline::Spline1D spx, spy;
    double s0 = 0.0, L = 0.0;
    std::vector<geom::Vec2> center;
    {
        timing::ScopedAcc _t("5) 스플라인+균일 재샘플", &T_spline);
        int paddingK = closed_mode ? 3 : 0;
        center = centerline::splineUniformClosed_EXPORT_CONTEXT(
            ordered, C.samples, paddingK, /*close_loop=*/C.emit_closed_duplicate,
            spx, spy, s0, L);
    }

    // [7] 센터라인 CSV 저장
    {
        timing::ScopedAcc _t("6) centerline 저장", &T_saveCenter);
        std::ofstream fo(outPath);
        if (!fo)
        {
            cerr << "[ERR] save centerline " << outPath << "\n";
            return 5;
        }
        fo.setf(std::ios::fixed);
        fo.precision(9);
        for (const auto &p : center)
            fo << p.x << "," << p.y << "\n";
    }

    // [8] 기하량 + 폭(width) 계산 및 저장  (재구성 링의 엣지를 사용)
    {
        timing::ScopedAcc _t("7) geom+width 계산/저장", &T_geom);

        // 재구성 링에서 엣지 구성
        std::vector<std::pair<geom::Vec2, geom::Vec2>> innerE, outerE;
        if (closed_mode){
            innerE = clip::ringEdges(inner_from_mids);
            outerE = clip::ringEdges(outer_from_mids);
        } else {
            innerE = clip::ringEdgesPolyline(inner_from_mids);
            outerE = clip::ringEdgesPolyline(outer_from_mids);
        }

        std::ofstream fo2(base + "_with_geom.csv");
        if (!fo2) { cerr << "[ERR] save centerline_with_geom\n"; return 6; }
        fo2.setf(std::ios::fixed);
        fo2.precision(9);

        fo2 << "s,x,y,heading_rad,curvature,dist_to_inner,dist_to_outer,width,v_kappa_mps\n";

        double x0=0, y0=0, hd0=0, k0=0, din0=0, dout0=0, w0=0, v0=0;
        const int Ncenter = (int)center.size();
        const int Kmax = closed_mode ? C.samples : Ncenter;
        const int denomN = closed_mode ? C.samples : std::max(1, C.samples);

        for (int k = 0; k < Kmax; ++k) {
            double si = s0 + L * (double(k) / double(denomN));
            double x, xp, xpp, y, yp, ypp;
            spx.eval_with_deriv(si, x, xp, xpp);
            spy.eval_with_deriv(si, y, yp, ypp);

            double heading = std::atan2(yp, xp);
            double speed2 = xp * xp + yp * yp;
            double denom = std::pow(std::max(1e-12, speed2), 1.5);
            double curvature = (xp * ypp - yp * xpp) / denom;

            geom::Vec2 nvec = geom::normalize(geom::Vec2{-yp, xp}, 1e-12);
            double d_in = std::numeric_limits<double>::infinity();
            double d_out = std::numeric_limits<double>::infinity();
            if (nvec.x != 0 || nvec.y != 0) {
                geom::Vec2 P{x, y};
                distancesToRings(P, nvec, innerE, outerE, d_in, d_out);
            }
            if (!std::isfinite(d_in))  d_in  = 0.0;
            if (!std::isfinite(d_out)) d_out = 0.0;

            double width = d_in + d_out;
            double si_rel = si - s0;

            double denom_k = std::max(std::fabs(curvature), C.kappa_eps);
            double v_kappa = std::sqrt(C.a_lat_max / denom_k);
            if (v_kappa > C.v_cap_mps) v_kappa = C.v_cap_mps;

            if (k == 0) {
                x0=x; y0=y; hd0=heading; k0=curvature; din0=d_in; dout0=d_out; w0=width; v0=v_kappa;
            }

            fo2 << si_rel << "," << x << "," << y << "," << heading << "," << curvature
                << "," << d_in << "," << d_out << "," << width << "," << v_kappa << "\n";
        }

        if (C.emit_closed_duplicate) {
            fo2 << L << "," << x0 << "," << y0 << "," << hd0 << "," << k0
                << "," << din0 << "," << dout0 << "," << w0 << "," << v0 << "\n";
        }
    }

    // [9] 최소 곡률 raceline (최적화) — 재구성 링 사용
    std::vector<geom::Vec2> center_for_opt = center;
    if (closed_mode &&
        center_for_opt.size() >= 2 &&
        geom::almostEq(center_for_opt.front(), center_for_opt.back(), 1e-12))
    {
        center_for_opt.pop_back();
    }
    raceline_min_curv::Result res;
    {
        timing::ScopedAcc _t("8) 최소곡률 레이싱라인 최적화", &T_race);

        std::vector<std::pair<geom::Vec2, geom::Vec2>> innerE, outerE;
        if (closed_mode){
            innerE = clip::ringEdges(inner_from_mids);
            outerE = clip::ringEdges(outer_from_mids);
        } else {
            innerE = clip::ringEdgesPolyline(inner_from_mids);
            outerE = clip::ringEdgesPolyline(outer_from_mids);
        }

        res = raceline_min_curv::compute_min_curvature_raceline(
            center_for_opt, innerE, outerE, C.veh_width_m, /*L=*/L, /*closed=*/closed_mode);
    }

    // [10] raceline 저장(좌표/geom)
    {
        timing::ScopedAcc _t("9) raceline 저장(좌표/geom)", &T_saveRace);
        // 좌표
        {
            std::ofstream fo(base + "_raceline.csv");
            if (!fo)
            {
                cerr << "[ERR] save raceline\n";
                return 7;
            }
            fo.setf(std::ios::fixed);
            fo.precision(9);
            const int Nrl = (int)res.raceline.size();
            for (int k = 0; k < Nrl; ++k)
                fo << res.raceline[k].x << "," << res.raceline[k].y << "\n";
            if (C.emit_closed_duplicate && Nrl > 0)
                fo << res.raceline[0].x << "," << res.raceline[0].y << "\n";
        }
        // with geom
        {
            std::ofstream fo(base + "_raceline_with_geom.csv");
            if (!fo) { cerr << "[ERR] save raceline_with_geom\n"; return 8; }
            fo.setf(std::ios::fixed);
            fo.precision(9);
            fo << "s,x,y,heading_rad,curvature,alpha_last,v_kappa_mps\n";

            const int Nrl = (int)res.raceline.size();
            for (int k = 0; k < Nrl; ++k) {
                double si = s0 + L * (double(k) / double(Nrl));
                double si_rel = si - s0;

                double denom_k = std::max(std::fabs(res.curvature[k]), C.kappa_eps);
                double v_kappa = std::sqrt(C.a_lat_max / denom_k);
                if (v_kappa > C.v_cap_mps) v_kappa = C.v_cap_mps;

                fo << si_rel << "," << res.raceline[k].x << "," << res.raceline[k].y << ","
                   << res.heading[k] << "," << res.curvature[k] << "," << res.alpha_last[k] << ","
                   << v_kappa << "\n";
            }
            if (C.emit_closed_duplicate && !res.raceline.empty()) {
                double denom_k0 = std::max(std::fabs(res.curvature[0]), C.kappa_eps);
                double v0 = std::sqrt(C.a_lat_max / denom_k0);
                if (v0 > C.v_cap_mps) v0 = C.v_cap_mps;

                fo << L << "," << res.raceline[0].x << "," << res.raceline[0].y << ","
                   << res.heading[0] << "," << res.curvature[0] << "," << res.alpha_last[0] << ","
                   << v0 << "\n";
            }
        }
    }

    // 요약 출력
    {
        auto total_ms = std::chrono::duration_cast<timing::Ms>(timing::Clock::now() - _t_all).count();
        std::cerr.setf(std::ios::fixed);
        std::cerr << "[TIME][SUMMARY] total=" << std::setprecision(3) << total_ms << " ms  |  "
                  << "load=" << T_load
                  << ", dt=" << T_dt
                  << ", mids=" << T_mids << ", mst=" << T_orderMST
                  << ", spline=" << T_spline << ", saveC=" << T_saveCenter
                  << ", geom=" << T_geom << ", race=" << T_race
                  << ", saveR=" << T_saveRace << "\n";
    }
    return 0;
}
