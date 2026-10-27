"""Fill template.html with the collected verification data. Usage: python build_report.py DATA_JSON OUT_HTML"""
import datetime
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.loads(Path(sys.argv[1]).read_text())
OUT = Path(sys.argv[2])
FINAL = {"heat_h": "heat_h_job93325", "conventional_h": "conventional_h_job93326", "heat_edge": "heat_edge_job93725", "heat_edge_mig": "heat_edge_job93319",
         "conventional_wse2": "conventional_wse2_job93323", "heat_wse2": "heat_wse2_job93327"}
A, RR = D["audit"], D["reruns"]


def chip(kind, text):
    return f'<span class="chip {kind}">{text}</span>'


OK, WARN, BAD, NA = (lambda t="Confirmed": chip("ok", t)), (lambda t="Discrepancy": chip("warn", t)), \
    (lambda t="Fails": chip("bad", t)), (lambda t="Not run": chip("na", t))


def f(v, d=4):
    return "–" if v is None else f"{v:,.{d}f}"


# ---------------------------------------------------------------- reruns
fr = {}
for key, name in FINAL.items():
    r = RR.get(name, {})
    fr[key] = r
hit = {}
for key in ("heat_h", "conventional_h", "heat_edge"):
    c = fr[key].get("curve") or []
    h = next((p for p in c if p["gap"] < 1e-4), None)
    hit[key] = dict(hit=h, last=c[-1] if c else None, gens=len(c), partial=fr[key].get("partial_generation_members"))
_gm = [r for r in fr["conventional_wse2"].get("generation_metrics", []) if int(r.get("members", 50)) >= 50]  # complete generations only
conv_wse2 = [dict(g=int(r["generation"]), loss=float(r["best_so_far_score"]), evals=0) for r in _gm]
ev = 0
for p, r in zip(conv_wse2, _gm):
    ev += int(float(r["model_energy_evaluations"]))
    p["evals"] = ev
fr["conventional_wse2_curve"] = conv_wse2
fr["heat_wse2_curve"] = fr["heat_wse2"].get("loss_curve") or []


def rerun_text(key, label):
    h = hit[key]
    if h["hit"]:
        p = h["hit"]
        return f"{label} reached its reference at generation {p['g']} ({p['evals']:,} evals, gap {p['gap']:+.1e} eV/atom)"
    if h["last"]:
        p = h["last"]
        return f"{label} has not reached its reference after {h['gens']} generations (best gap {p['gap']:.4f} eV/atom, {p['evals']:,} evals)"
    return f"{label} had not completed a generation when this report was built"


parts = [rerun_text("heat_h", "HEAT H"), rerun_text("conventional_h", "conventional H")]
if hit["heat_edge"]["last"]:
    parts.append(rerun_text("heat_edge", "HEAT edge (A100)"))
else:
    parts.append("the HEAT edge rerun ran out of GPU memory on a 20 GB MIG slice in its first crossover generation, hung instead of failing (issue below), and is queued again on a full A100")
hw = fr["heat_wse2"].get("loss_curve") or []
if hw:
    parts.append(f"a reduced HEAT WSe<sub>2</sub> run (population 10, otherwise production settings) went from {hw[0]['loss']:.3f} to {hw[-1]['loss']:.3f} in {len(hw)} generations, "
                 f"in line with the archive's {D['recorded']['wse2']['eat'][0]['loss']:.3f} → {D['recorded']['wse2']['eat'][1]['loss']:.3f} at population 50")
if conv_wse2:
    parts.append(f"the conventional WSe<sub>2</sub> driver, with its YAML fixed, completed {len(conv_wse2)} generation(s) (best loss {conv_wse2[-1]['loss']:.4f}; archive at the same point: "
                 f"{D['recorded']['wse2']['ordinary'][len(conv_wse2) - 1]['loss']:.4f})")
rerun_verdict = ("<strong>Fresh searches with the real model on our cluster:</strong> " + "; ".join(parts) +
                 ". A full HEAT WSe<sub>2</sub> search takes about five GPU-days in the archive, so we did not rerun it; it is covered by rescoring and the recorded metrics instead.")

# ---------------------------------------------------------------- claims
H, E, W = A["H_sublattice"], A["edge"], A["wse2"]
he, ho = H["eat"], H["ordinary"]
ee, eo = E["eat"], E["ordinary"]
we, wo = W["eat"], W["ordinary"]
bm = we["budget_matched"]["heat_best_through_gen_43"]
rs = D["rescore"]["R2_WSe2_full_loss"]
geo = D["wse2_free"]["D_geometry"]["heat_g128_crop"]
C = D["wse2_free"]
hh = hit["heat_h"]["hit"]
he_r = hit["heat_edge"]["hit"]
claims = [
    ("H: HEAT reaches the relaxed reference at generation 2", "gen 2", f"gen {he['hit']['generation']}; gap {he['hit']['gap_eV_per_atom']:+.1e} eV/atom, 240 H",
     "archive log", OK()),
    ("H: …using 24,000 model evaluations and 0.5 h", "24,000 · 0.5 h", f"{he['hit']['cumulative_evaluations']:,} · {he['hit']['cumulative_wall_hours']:.2f} h", "archive log", OK()),
    ("H: HEAT reproduces this in a fresh run", "–", rerun_text("heat_h", "rerun").replace("rerun ", "") if hh or hit['heat_h']['last'] else "in progress",
     "rerun, real UMA, MP refs", OK("Reproduced") if hh else WARN("Not yet") if hit['heat_h']['last'] else NA("Running")),
    ("H: conventional GA at generation 20", "0.012 eV/H (2.9 eV total)",
     f"{ho['at_generation_20']['gap_eV_per_atom']:.4f} eV/atom = {ho['at_generation_20']['gap_eV_per_H_atom']:.4f} eV/H = {ho['at_generation_20']['total_energy_gap_eV']:.2f} eV per 330-atom cell",
     "archive log; plot script’s own per-H formula", WARN("Unit slip")),
    ("H: conventional GA in a fresh run, generation 20", "0.012 (per atom)", rerun_text("conventional_h", "rerun").replace("rerun ", ""), "rerun, real UMA, MP refs",
     OK("Reproduced") if hit["conventional_h"]["last"] and not hit["conventional_h"]["hit"] and hit["conventional_h"]["gens"] >= 21 else NA("Running")),
    ("H: …after 91,000 evaluations and 1.6 h", "91,000 · 1.6 h", f"{ho['at_generation_20']['cumulative_evaluations']:,} · {ho['at_generation_20']['cumulative_wall_hours']:.2f} h", "archive log", OK()),
    ("H: conventional GA does not reach the reference", "not within 20 gens", f"reaches it at gen {ho['hit']['generation']} ({ho['hit']['cumulative_evaluations']:,} evals, {ho['hit']['cumulative_wall_hours']:.1f} h)",
     "archive log (100 gens)", WARN("Window-dependent")),
    ("H: HEAT slightly less time per generation", "slightly less", f"{he['per_generation_wall_s_matched_window']:.0f} s vs {ho['per_generation_wall_s_matched_window']:.0f} s (gens 1–20)", "archive log", OK()),
    ("Edge: HEAT reaches the reference at generation 4", "gen 4 · 50,629 · 1.93 h", f"gen {ee['hit']['generation']} · {ee['hit']['cumulative_evaluations']:,} · {ee['hit']['cumulative_wall_hours']:.2f} h", "archive log", OK()),
    ("Edge: reference structure and energy", "E<sub>f</sub> = 0.117481", f"{fr['heat_edge'].get('reference_formation_energy', float('nan')):.6f} (E = {fr['heat_edge'].get('reference_total_energy', float('nan')):.3f} vs −7806.832 eV)",
     "rerun reference relaxation", OK("Reproduced")),
    ("Edge: HEAT reproduces this in a fresh run", "–", rerun_text("heat_edge", "rerun").replace("rerun ", "") , "rerun, real UMA",
     OK("Reproduced") if he_r else WARN("Not yet") if hit['heat_edge']['last'] else NA("Running")),
    ("Edge: conventional GA at generation 20", "0.096 eV/core atom · 160,434 · 5.23 h",
     f"{eo['at_generation_20']['gap_eV_per_core_atom']:.4f} · {eo['at_generation_20']['cumulative_evaluations']:,} · {eo['at_generation_20']['cumulative_wall_hours']:.2f} h; 0.084 with today’s MP Fe",
     "archive log", OK()),
    ("Edge: about the same time per generation", "≈ same", f"{ee['per_generation_wall_s_matched_window']:.0f} s vs {eo['per_generation_wall_s_matched_window']:.0f} s (gens 1–7)", "archive log", OK()),
    ("WSe<sub>2</sub>: HEAT 0.11 → −0.35, first at gen 115", "0.11 → −0.35 @115", f"{we['generation0_best']:.4f} → {we['at_128']['best_loss']:.4f}, first ≤ −0.35 at gen {we['first_generation_loss_le_m0p35']}", "archive metrics", OK()),
    ("WSe<sub>2</sub>: conventional 0.03 at gen 128", "0.03", f"{wo['at_128']['best_loss']:.4f} (0.0199 by gen 446)", "archive metrics", OK()),
    ("WSe<sub>2</sub>: E<sub>f</sub> −0.08 / −0.39 eV/atom; image 0.106 / 0.035", "", f"{wo['at_128']['formation_energy']:.3f} / {we['at_128']['formation_energy']:.3f}; {wo['at_128']['weighted_image_penalty']:.3f} / {we['at_128']['weighted_image_penalty']:.3f}", "archive metrics", OK()),
    ("WSe<sub>2</sub>: HEAT driver runs with the real model", "–", (f"population 10: {hw[0]['loss']:.4f} → {hw[-1]['loss']:.4f} over {len(hw)} generations" if hw else "running"),
     "rerun, reduced population", OK("Runs") if hw else NA("Running")),
    ("WSe<sub>2</sub>: conventional driver, generation 0", "0.1569 (archive)", (f"{conv_wse2[0]['loss']:.4f} with {conv_wse2[0]['evals']:,} evals (archive 4,610)" if conv_wse2 else "running"),
     "rerun, production settings, YAML fixed", OK("Consistent") if conv_wse2 else NA("Running")),
    ("WSe<sub>2</sub>: guided reference loss −0.3567", "−0.3567", f"{rs['final_guided_relaxed']['score']:.6f} (Δ {rs['final_guided_relaxed']['abs_diff_score']:.1e})", "rescored, real UMA", OK("Reproduced")),
    ("WSe<sub>2</sub>: ideal bilayer −0.42 eV/atom", "−0.42", f"{rs['previous_energy_only_relaxed']['formation_energy']:.5f} (Δ {rs['previous_energy_only_relaxed']['abs_diff_formation']:.1e})", "rescored, real UMA", OK("Reproduced")),
    ("WSe<sub>2</sub>: 540,000 vs 210,000 evaluations", "2.6×", f"{we['at_128']['cumulative_evaluations']:,} vs {wo['at_128']['cumulative_evaluations']:,} ({we['at_128']['cumulative_evaluations'] / wo['at_128']['cumulative_evaluations']:.2f}×)", "archive metrics", OK()),
    ("WSe<sub>2</sub>: at 210k evaluations (gen 43) HEAT has −0.254 eV/atom, image 0.0615", "", f"{bm['formation_energy']:.4f}, {bm['weighted_image_penalty']:.4f} ({bm['cumulative_evaluations']:,} evals)", "archive members", OK()),
    ("WSe<sub>2</sub>: 1497 atoms; crop 1150 (389 W, 761 Se)", "", f"{we['at_128']['atoms']}; {geo['atoms']} ({geo['W']} W, {geo['Se']} Se)", "metrics + bundled crop", OK()),
    ("WSe<sub>2</sub>: six atomic planes of two twisted trilayers", "", f"6 pure planes; {100 * geo['W_with_3_Se_above_and_3_below']:.0f}% of W 3+3-coordinated; twist {geo['twist_between_W_layers_deg']:.2f}° (guided ref −2.89°)", "geometry of bundled crop", OK()),
    ("WSe<sub>2</sub>: projection Pearson 0.934 vs 0.124", "0.934 / 0.124", f"{C['C_heat']['pearson_bundled_tif_vs_experiment']:.4f} / {C['C_ordinary']['pearson_bundled_tif_vs_experiment']:.4f}", "recomputed from bundled TIFFs + structures", OK()),
]
claim_rows = "".join(f"<tr><td>{a}</td><td class='num'>{b}</td><td class='num' style='white-space:normal'>{c}</td><td>{d}</td><td>{e}</td></tr>" for a, b, c, d, e in claims)

# ---------------------------------------------------------------- issues
issues = [
    (BAD("Blocker"), "<code>run_search.py</code> cannot start three of the six searches",
     "The launcher creates the run directory to write <code>portable_config.yaml</code>, then the H-sublattice HEAT driver and both conventional fixed-scaffold drivers abort with "
     "“Output directory exists”. <code>heat edge</code> and <code>heat wse2</code> get past this point. Fix: write the YAML next to the run directory (proposed patch in <code>verification/proposed_fixes.patch</code>)."),
    (BAD("Blocker"), "Conventional WSe<sub>2</sub> config is read as a string",
     "<code>transform_epsilon: 1e-06</code> in <code>discrete_09.30.26_wse2.yaml</code> is a string under PyYAML (YAML 1.1 needs a decimal point), so config validation crashes. "
     "This breaks <code>run_search.py conventional wse2</code> and the driver’s own unit tests. The source snapshot in <code>baseline_ptycho_guided_320/source/</code> has the working <code>1.0e-06</code>; "
     "the launcher’s <code>json.dumps</code> re-introduces the problem for any such float."),
    (BAD("Robustness"), "A failed worker hangs the search instead of stopping it",
     "When a worker raises (for us: CUDA out of memory on a 20 GB GPU while relaxing an edge crossover child, whose superposition roughly doubles the active sites), "
     "the main process leaves the generation loop and its <code>finally: stop_ga_workers</code> joins workers that are blocked writing results into a queue nobody drains. "
     "The job sat idle for 8 h with the traceback hidden; killing the workers surfaced it. Draining or terminating workers in the <code>finally</code> block would make failures visible. "
     "Separately, plan for more than 20 GB of GPU memory per worker for the edge case (the archive used larger consumer GPUs)."),
    (WARN("Text"), "H-sublattice gap is reported per atom but labelled per H",
     f"At generation 20 the conventional GA is {ho['at_generation_20']['gap_eV_per_atom']:.4f} eV/atom above the reference, which is {ho['at_generation_20']['gap_eV_per_H_atom']:.4f} eV per H and "
     f"{ho['at_generation_20']['total_energy_gap_eV']:.2f} eV per 330-atom cell. The paper says 0.012 eV per H (2.9 eV total). Fig. 2b, which uses the plot script’s per-H conversion, should end near 0.016."),
    (WARN("Framing"), "The conventional GA does reach the H reference, later",
     f"The archived 100-generation run reaches the reference at generation {ho['hit']['generation']} after {ho['hit']['cumulative_evaluations']:,} evaluations ({ho['hit']['cumulative_wall_hours']:.1f} h), "
     "about 7× HEAT’s cost. The Results text is correct for the 20-generation window. The Discussion’s “fails to reach them over a longer search” fits the edge case but not H."),
    (WARN("Provenance"), "Elemental references are not pinned",
     "Formation energies use Materials Project structures fetched at run time. W and Se from today’s MP match the archive exactly. Fe does not: the archived edge run used μ<sub>Fe</sub> = −8.1741 eV/atom, while today’s mp-13 gives −8.2674. "
     f"For Mg/B/H, the same relaxed H reference gives E<sub>f</sub> = {fr['heat_h'].get('reference_formation_energy', float('nan')):.5f} today against the hard-coded −0.22290. "
     "Gaps at fixed composition are unaffected, but the absolute values cannot be regenerated. Bundling a <code>reference_energies.json</code> per case, as already done for WSe<sub>2</sub>, would fix this."),
    (WARN("Missing input"), "One WSe<sub>2</sub> unit test needs an unbundled TIFF",
     "<code>test_slice_count_validation_and_30_slice_support</code> reads <code>chia-hao_data/objp_zstack_iter0001_30_slices.tif</code>, which is not in the archive. The other 24 bundled tests pass, apart from the YAML-blocked suite above."),
    (WARN("Scope"), "Single stochastic run per method; structure exports removed",
     "Each comparison is one run per method, as the README says, so the generation counts are single samples. Our reruns add one more sample each. Per-generation XSF exports were removed, "
     "so the Pearson-vs-generation curve and the structure renders cannot be regenerated. The final structures that the figures use are still bundled."),
    (NA("Note"), "Many final relaxations stop at the step cap",
     f"{100 * ee['final_relax_unconverged_fraction']:.0f}% of HEAT edge members and {100 * he['final_relax_unconverged_fraction']:.0f}% of HEAT H members end their final FIRE relaxation at 500 steps without reaching f<sub>max</sub> = 0.05 eV/Å, "
     "and are scored as they are (the conventional logs do not record this). The reference-hitting candidates are converged, so the conclusions stand, but unconverged scores are biased high."),
]
issue_html = "".join(f'<div class="issue">{c}<h3>{t}</h3><p>{b}</p></div>' for c, t, b in issues)

# ---------------------------------------------------------------- health + matrix + cost
tests = D["repo_tests"]
n_ok = sum(v["ran"] for v in tests.values() if v["status"] == "OK")
n_all = sum(v["ran"] for v in tests.values())
health = [
    ("Bundled unit tests (WSe<sub>2</sub> drivers)", f"{n_ok} of {n_all} collected tests pass; <code>test_gaussian_09_30</code> 7/8 (missing TIFF); <code>test_discrete_wse2</code> cannot start (YAML bug)", WARN("Partial")),
    ("Documented plot reproduction", "All three scripts run and reproduce the recorded curves and endpoint numbers", OK("Pass")),
    ("<code>run_search.py --prepare-only</code>", "6 of 6 method × case combinations", OK("Pass")),
    ("Portable configs vs recorded run parameters", "0 value differences across about 400 shared settings; population splits match the paper", OK("Pass")),
    ("FAIRChem EAT patch", "Applies cleanly to <code>cce3699</code>; with no effective-atom weights, results are identical to stock FAIRChem", OK("Pass")),
    ("Recorded energies with today’s software", "μ<sub>W</sub>, μ<sub>Se</sub> to 2×10<sup>−7</sup> eV/atom; three WSe<sub>2</sub> losses to ≤ 4×10<sup>−7</sup>; edge reference to 1.2 meV total", OK("Pass")),
]
health_rows = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in health)
L = D["launcher"]


def launch(key):
    e = L.get(key, "")
    if "Output directory exists" in e:
        return BAD("Dir exists")
    if "isfinite" in e:
        return BAD("YAML bug")
    if "MP" in e:
        return OK("Starts")
    return NA("?")


def runstat(key, label_missing):
    if key == "heat_wse2":
        return OK(f"pop 10, {len(hw)} gens") if hw else NA("Running")
    if key == "heat_edge" and not hit["heat_edge"]["last"]:
        return BAD("OOM on MIG") + " " + NA("A100 queued")
    if key == "conventional_wse2":
        return OK(f"{len(conv_wse2)} gen{'s' if len(conv_wse2) != 1 else ''}, YAML fixed") if conv_wse2 else NA("Running")
    h = hit.get(key)
    if not h or not h["last"]:
        return NA("Running")
    return OK(f"Hit at gen {h['hit']['g']}") if h["hit"] else WARN(f"{h['gens']} gens, no hit yet")


cl = D["config_loading"]
matrix = []
for m, c, rk, miss in [("heat", "h", "heat_h", ""), ("heat", "edge", "heat_edge", ""), ("heat", "wse2", "heat_wse2", ""),
                       ("conventional", "h", "conventional_h", ""), ("conventional", "edge", None, "> 24 h on MIG; archive log used"),
                       ("conventional", "wse2", "conventional_wse2", "")]:
    st = cl[f"{m}_{c}"]["status"]
    matrix.append(f"<tr><td>{m} · {c}</td><td>{OK('0 diffs')}</td><td>{OK('Loads') if st == 'loaded' else BAD('Crashes')}</td><td>{launch(f'{m}-{c}')}</td>"
                  f"<td>{runstat(rk, miss) if rk else NA(miss)}</td></tr>")
matrix_rows = "".join(matrix)
wh = lambda m: D["recorded"]["wse2"][m][-1]["hours"]  # noqa: E731
cost = [
    ("HEAT · H", "0–2 (to reference)", f"{he['hit']['cumulative_evaluations']:,}", f"{he['hit']['cumulative_wall_hours']:.1f} h (2 GPUs)", OK("No")),
    ("Conventional · H", "0–20", f"{ho['at_generation_20']['cumulative_evaluations']:,}", f"{ho['at_generation_20']['cumulative_wall_hours']:.1f} h (2 GPUs)", OK("No")),
    ("HEAT · edge", "0–4 (to reference)", f"{ee['hit']['cumulative_evaluations']:,}", f"{ee['hit']['cumulative_wall_hours']:.1f} h (2 GPUs)", OK("No")),
    ("Conventional · edge", "0–20", f"{eo['at_generation_20']['cumulative_evaluations']:,}", f"{eo['at_generation_20']['cumulative_wall_hours']:.1f} h (2 GPUs)", OK("No")),
    ("HEAT · WSe<sub>2</sub>", "0–128", f"{we['at_128']['cumulative_evaluations']:,}", f"≈ 5.1 days from log timestamps ({wh('eat'):.0f} member-hours)", BAD("Yes")),
    ("Conventional · WSe<sub>2</sub>", "0–128 (0–446)", f"{wo['at_128']['cumulative_evaluations']:,} ({wo['total_evaluations']:,})", f"{wh('ordinary'):.0f} member-hours to gen 446", WARN("Likely")),
]
cost_rows = "".join(f"<tr><td>{a}</td><td class='num'>{b}</td><td class='num'>{c}</td><td>{d}</td><td>{e}</td></tr>" for a, b, c, d, e in cost)

# ---------------------------------------------------------------- EAT rows
er = D["eat_real"]
eat_rows = "".join(f"<tr><td>{a}</td><td>{b}</td><td class='num' style='white-space:normal'>{c}</td><td>{d}</td></tr>" for a, b, c, d in [
    ("Pure weights reproduce ordinary atoms", "one-hot <code>eat_weights</code> vs integer species, Fe<sub>54</sub> and Mg(BH<sub>4</sub>)<sub>2</sub>",
     f"ΔE ≤ {max(er['T1_Fe54']['dE'], er['T1_MgB2H8']['dE']):.1e} eV, ΔF ≤ {max(er['T1_Fe54']['dF_max'], er['T1_MgB2H8']['dF_max']):.1e} eV/Å", OK("Pass")),
    ("Patch leaves normal use unchanged", "patched vs stock FAIRChem, same structures", f"ΔE ≤ {max(er['T2_Fe54']['dE'], er['T2_MgB2H8']['dE']):.1e} eV", OK("Pass")),
    ("Analytic dE/dw is correct", "autograd vs central finite differences on fractional sites",
     f"≤ {100 * max(er['T3_Fe54']['max_rel_err'], er['T3_MgB2H8']['max_rel_err']):.2f}% per component; directional {100 * max(er['T3_Fe54']['directional_rel_err'], er['T3_MgB2H8']['directional_rel_err']):.2f}%", OK("Pass")),
    ("A null site is absent", "add a site with all weights 0",
     f"ΔE ≤ {max(er['T4_Fe54']['null_H_symbol']['dE'], er['T4_MgB2H8']['null_H_symbol']['dE'], er['T4_Fe54']['null_Fe_symbol']['dE'], er['T4_MgB2H8']['null_Fe_symbol']['dE']):.1e} eV, ΔF on real atoms ≤ {max(er['T4_Fe54']['null_H_symbol']['dF_real_max'], er['T4_MgB2H8']['null_H_symbol']['dF_real_max']):.1e} eV/Å; turns on smoothly with occupancy", OK("Pass")),
    ("Drivers call it correctly", "H driver’s weight layout and gradient helpers vs direct calls",
     f"ΔE {er['T5_driver_helpers']['dE']:.1e} eV, Δgrad {er['T5_driver_helpers']['dGrad_max']:.1e}", OK("Pass")),
])

# ---------------------------------------------------------------- captions + data
def cap(key, recorded_hit, label):
    h = hit[key]
    s = f"Archive: HEAT reaches the reference at generation {recorded_hit}. "
    if h["last"]:
        s += rerun_text(key, f"Our {label} rerun").replace("Our", "Our") + ". "
    if key == "heat_h" and hit["conventional_h"]["last"]:
        s += rerun_text("conventional_h", "Our conventional rerun") + "."
    return s


captions = dict(h=cap("heat_h", he["hit"]["generation"], "HEAT"), edge=cap("heat_edge", ee["hit"]["generation"], "HEAT")
                + " The conventional edge search was not rerun (more than 24 h on our MIG slices).")
payload = dict(recorded=D["recorded"], wse2_free={"D_geometry": D["wse2_free"]["D_geometry"]},
               final_reruns={k: {kk: vv for kk, vv in v.items() if kk not in ("summary",)} if isinstance(v, dict) else v for k, v in fr.items()},
               captions=captions, eat_rows=eat_rows)
html = (HERE / "template.html").read_text()
for k, v in {"__DATE__": datetime.date.today().isoformat(), "__RERUN_VERDICT__": rerun_verdict, "__CLAIM_ROWS__": claim_rows,
             "__ISSUES__": issue_html, "__HEALTH_ROWS__": health_rows, "__MATRIX_ROWS__": matrix_rows, "__COST_ROWS__": cost_rows,
             "__DATA__": json.dumps(payload, default=float).replace("</", "<\\/")}.items():
    html = html.replace(k, v)
OUT.write_text(html)
print("WROTE", OUT, len(html) // 1024, "KB")
