# 🏎️ Formula Student Driverless — DT 기반 센터라인 & 최소곡률/최소시간 레이싱라인

이 프로젝트는 FSDS 등에서 수집한 **내/외측 콘(inner/outer) 좌표**를 입력으로 받아,
**Delaunay Triangulation → 경계 중점(MST 정렬) → 스플라인 균일 재샘플 → 폭/기하량 계산 → 최소곡률/최소시간 레이싱라인 최적화**
를 한 번에 수행합니다. 핵심 철학은 **단일 파이프라인 + 풍부한 CSV 디버그 덤프 + 튜닝 친화성**입니다.

---

## ✨ 주요 기능
- **DT(Bowyer–Watson)만 사용**: CDT/클리핑/품질필터 없이 순수 Delaunay로 경량·견고 구현
- **라벨-상이한 경계 엣지 추출 + 길이 기반 필터**(중앙값 스케일/절대 상한)
- **MST 지름 경로 기반 중점 순서화**(k-NN 그래프) + **오픈/폐루프 방향·시작점 정합**
- **자연 3차 스플라인 + 균일 arc-length 재샘플**
- **법선 레이캐스팅으로 폭(width) 계산**: centerline에서 inner/outer까지 거리
- **최소곡률(min-curv) 레이싱라인**: 곡률 선형화 + 투영 경사강하(PGD, Armijo, box 제약)
- **최소시간(min-time) 레이싱라인**:
  v(s) 전·후진 패스(마찰원 + 공력·구름저항 + 파워리밋), corner/slow 가중으로 time-weighted min-curv
- **풍부한 CSV 덤프**: 중간 산출물/속도·가속/랩타임 비교까지 전 단계 로그화
- **오픈/폐루프 모두 지원**: 현재 위치/헤딩 또는 시작 앵커/헤딩으로 자동 정렬

---

## 🧱 입력 & 출력
### 입력 파일
- inner.csv, outer.csv  (각 줄: `x,y`, 단위 m, 헤더 불필요)

### 빌드 & 실행
```bash
g++ -std=c++17 -O3 main.cpp -o fsd_path
./fsd_path inner.csv outer.csv centerline.csv
```
- 세 번째 인자 `centerline.csv`는 **최종 센터라인** 저장 경로이며,
  확장자를 뗀 **베이스 이름**으로 각종 디버그 CSV가 함께 생성됩니다.

### 주요 출력 파일(접두사: centerline → centerline_*) 
- `centerline.csv` : 최종 센터라인 좌표
- `centerline_with_geom.csv` : s,x,y,heading_rad,curvature,dist_to_inner,dist_to_outer,width,v_kappa_mps
- `*_raceline.csv`, `*_raceline_with_geom.csv`(최소곡률) : s,x,y,heading_rad,curvature,alpha_last,v_kappa_mps
- `*_mintime_raceline.csv`, `*_mintime_with_geom.csv`(최소시간) : s,x,y,heading_rad,curvature,alpha_last,v_mps,ax_mps2
  → 실행 로그에 **예상 랩타임** 출력
- 디버그/중간 산출물:
  - `*_all_points.csv`(id,x,y,label), `*_tri_raw_idx.csv`(a,b,c)
  - `*_edges_all_idx.csv`, `*_edges_labeldiff_idx.csv`, `*_edges_labeldiff_kept_idx.csv`
  - `*_mids_raw.csv`, `*_mids_ordered.csv`
  - `*_inner_from_mids.csv`, `*_outer_from_mids.csv`
  - `*_debug_compare_paths.csv` : 센터/최소곡률/최소시간 좌표·편차·κ·v·ax·가중 비교
- 폐루프 + `emit_closed_duplicate=true`이면 s=L에서 첫 점 재기록(클로즈드 편의용).

---

## 🧭 파이프라인 개요
1) Delaunay(Bowyer–Watson) 생성 → 엣지맵 구성
2) 라벨-상이한 경계 엣지 추출 → **길이 기반 필터** 적용
3) 경계 엣지 **중점(mids)** 계산 → **MST 지름 경로**로 순서화
4) **방향/시작점 정합**
   - 오픈: 현재 위치/헤딩과 정렬, 시작 앵커에 가장 가까운 점을 0번으로 회전
   - 폐루프: 시작 앵커/헤딩(또는 CCW 강제) 정합 후 시작점 회전
5) **자연 3차 스플라인 + 균일 재샘플**(패딩으로 경계 안정화)
6) **폭/기하량 계산**: 법선 레이캐스트로 inner/outer까지 거리 → 폭, v_κ
7) **최소곡률 레이싱라인**: κ 선형화, PGD+Armijo, 코리도(box) 제약
8) **최소시간 레이싱라인**: v(s) 전·후진 패스(마찰원, 항력/구름, 파워리밋) + corner/slow 가중 → 반복

---

## ⚙️ 설정(튜닝) — cfg::Config 주요 항목
- 트랙/정렬: `is_closed_track`, `force_closed_ccw`, `current_pos_*`, `current_heading_rad`,
  `start_anchor_*`, `start_heading_rad`, `emit_closed_duplicate`
- 샘플링/정렬: `use_dynamic_samples`, `sample_factor_n`, `samples_min/max`, `knn_k`
- 경계 엣지 필터: `enable_boundary_len_filter`, `boundary_edge_len_scale`, `boundary_edge_abs_max`
- 차량/안전: `veh_width_m`, `safety_margin_m`
- 최적화: `lambda_smooth`, `max_outer_iters`, `max_inner_iters`, `step_init`, `step_min`, `armijo_c`
- 제한/동역학: `kappa_eps`, `v_cap_mps`, `mu`(→`a_total_max`), `a_lat_max`,
  `a_long_acc_cap`, `a_long_brake_cap`, `Cd`, `A_front_m2`, `rho_air`, `c_rr`, `P_max_W`, `mass_kg`
- 최소시간 가중: `w_time_gain`, `time_gamma_power`, `time_weight_use_inv_v`, `inv_v_gain`,
  `max_vpass_iters`, `use_total_ge_lat`
- 디버그: `verbose`, `debug_dump`, `debug_offset_warn_m`

### 튜닝 힌트
- 코너 공략↑: `w_time_gain↑` 또는 `lambda_smooth↓`(과도 시 지그재그 주의)
- 직선 가속 현실화: `P_max_W↓` 또는 `a_long_acc_cap↓`, `Cd·A`, `c_rr` 보정
- 경계 과접근 방지: `safety_margin_m↑`

---

## 🔎 디버깅 & 검증
- 단계별 **타이밍 로그**: load/dt/mids/mst/spline/geom/mincurv/mintime
- `*_debug_compare_paths.csv`:
  - 센터/최소곡률/최소시간의 좌표, **부호 있는 편차**(센터라인 법선 기준), κ, v, ax, 가중치 등
  - kappa-limited 비율, grad norm, Armijo 백트랙 횟수 로그

---

## 📐 알고리즘 노트
- **DT만 사용**: 두 링 가정 하에, 라벨-상이한 경계엣지의 **중점**이 트랙 중앙 경향 반영
- **폭 계산**: 스플라인 재샘플 좌표에서 법선 ± 방향으로 링 세그먼트에 **ray–segment** 교차 → 최단거리
- **최소곡률**: 유한차분 D1/D2 및 전치(D1ᵀ/D2ᵀ)로 그라디언트 구성 → PGD + Armijo
- **최소시간**:
  - `v ≤ √(a_lat_max / |κ|)` + 마찰원에서 종가속 여유 산출
  - 전진: `v_{i+1} ≤ √(v_i² + 2 a_acc h)` / 후진: `v_i ≤ √(v_{i+1}² + 2 a_brk h)`
  - **항력/구름저항** 및 **파워리밋** 반영으로 현실성 강화

---

## ⚠️ 전제 & 한계
- 입력 콘이 **대략 두 링(inner/outer)** 을 이룬다는 가정
- **CDT/클리핑 없음**: 외곽 이상치가 많으면 경계 엣지 필터 파라미터 튜닝 필요
- 폐루프 완전 스냅(첫·끝 정합)이 필요하면 후처리 스냅 옵션 추가 권장

---

## 🗺️ 로드맵(제안)
- 코리도 갱신 **EMA/클립**으로 수렴 안정성 추가
- 폐루프 **주기(cyclic) 스플라인** 옵션
- `orderIndicesByMST` 매칭에 **k-d tree** 도입(대규모 트랙 가속)
- 경계 근접 **가중 페널티 항**(안전 여유 반영)

---

## 📄 라이선스
연구/학습 목적 예제입니다. 실제 차량 적용 시 안전 확보 및 파라미터 검증이 필수입니다.

---

## 🙌 문의
PR/이슈로 제안/피드백 환영합니다. FSDS/실트랙 로그에 맞춘 튜닝 가이드도 환영합니다.
