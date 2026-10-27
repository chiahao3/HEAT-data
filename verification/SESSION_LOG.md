# HEAT-data verification — session log

Branch: `claude/verify-repo` (from `main` @ 385139c)
Started: 2026-10-26

## Log
- Created branch `claude/verify-repo`; surveyed repo layout (README, run_search.py, configs/, scripts/, images/).
- Downloaded arXiv:2610.28996 (v1) to ~/scratch/heat_verify/paper; extracted text. Key claims recorded (see report).
- Paper claims extracted (H: ref by gen 2, 24k evals/0.5 h; edge: ref by gen 4, 50,629 evals/1.93 h; WSe2: loss -0.35 vs 0.03, Pearson 0.934 vs 0.124).
- `run_search.py --prepare-only` succeeds for all 6 method×case combos.
- `verification/scripts/compare_configs.py`: configs/*.json match parameters/*/run_parameters.json with 0 value differences.
- Cluster: altas reachable; workspace is the same sshfs-mounted tree. No HF token / cached UMA weights anywhere; facebook/UMA is gated (HTTP 401).
- **User decision: weights-free verification only** (no UMA-dependent runs).
- fairchem cloned at cce3699 (2025-11-17) on altas; EAT patch `fairchem_local_changes.patch` applies cleanly. Building conda env `heat` (py3.12, torch 2.8.0+cu126, patched fairchem-core -e).
- Env `heat` on altas: torch 2.8.0+cu126, numpy 2.2.6, ase 3.29, patched fairchem-core (editable). Needed `setuptools<81` (torchtnt imports pkg_resources).
- Slurm wrapper `verification/slurm/{heat_job.sub,submit.sh}`: refuses >4 heat-* jobs; hides GPUs in CPU jobs (nodes are not cgroup-restricted: a CPU job saw all 4 A100s).
- Job 93296 (repo tests + replots): 24/26 repo unit tests pass. Failures: (1) `test_gaussian_09_30.test_slice_count_validation_and_30_slice_support` needs an unbundled TIFF (`chia-hao_data/objp_zstack_iter0001_30_slices.tif`); (2) `test_discrete_wse2` setUpClass crashes: `discrete_09.30.26_wse2.yaml` has `transform_epsilon: 1e-06`, which PyYAML reads as a *string* (original source copy has `1.0e-06`). All 3 plot scripts run and reproduce printed numbers.
- Config loading (job 93305): heat_h/edge/wse2 + conventional_h/edge load through their drivers with 0 mismatches; **conventional_wse2 fails** with the same YAML-typing bug -> `run_search.py conventional wse2` cannot launch as shipped.
- WSe2 weights-free (job 93301): preprocessing reproduces bundled 12x280x280 field exactly; image/box/composition terms of guided ref + conventional winner reproduced to ~1e-7; Fig.4c Pearson 0.9344/0.1241 reproduced from bundled TIFFs (0.9331/0.1233 re-rendered from cropped XSFs). Geometry: HEAT g128 crop has 6 planes (Se/W/Se x2), 74.6% of W with 3+3 Se trilayer coordination (conv 0.6%, HEAT g0 1.0%, guided ref 86.8%); W-layer twist -2.888 deg vs guided ref -2.890 deg.
- Recorded-results audit (job 93303): all H/edge/WSe2 numbers in paper reproduced from logs except: H conventional gap at gen 20 is 0.0163 eV/H (3.90 eV / 330-atom cell), paper says 0.012 eV/H (2.9 eV) = per-atom value mislabeled; H conventional wall 1.55 h (paper 1.6 h). Conventional H reaches the reference at gen 47 (170,386 evals) in the 100-gen log (paper only shows 20 gens). HEAT edge: 46% of final FIRE relaxations hit the 500-step cap (H: 12.5%); hit members are converged.
- EAT patch, fake UMA-architecture model (job 93306): one-hot == integer path (dE<=1.5e-5 eV); patched == stock fairchem bit-identical; dE/dw matches FD; driver helpers consistent. Null site NOT exactly invisible: dE 0.01-0.08 eV with fake model's max_neighbors=20, 0.04-0.06 eV (forces <=0.015 eV/A) with cap 300 -> re-test on real UMA.
- User provided HF access on workstation (22:27). Downloaded uma-s-1p1 (+iso_atom_elem_refs.yaml) locally (no GPU), rsync'd model cache only (no token) to altas `~/heat_verify/hf_cache`; jobs use FAIRCHEM_CACHE_DIR + HF_HUB_OFFLINE=1. User: all compute must stay on altas.
- No MP API key: elemental mu for W/Se from exact bundled MP structures; Mg/B/H/Fe served by an MP stub (sitecustomize) with UMA-relaxed substitute structures. mu_Fe implied by edge log = -8.174070 eV/atom.
