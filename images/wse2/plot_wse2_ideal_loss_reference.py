"""Plot total loss with a hypothetical zero-penalty energy reference.

The line uses the relaxed baseline's formation energy, setting image,
composition, confinement, and entropy contributions to zero. It is neither
the measured total loss of that structure nor a proven global lower bound.
Run with python3; the original total-loss PNG is preserved.
"""
import json
import plot_wse2_convergence as plotting


def main():
    heat = plotting.best_candidates_so_far(plotting.load_results(plotting.HEAT_CSV))
    conventional = plotting.best_candidates_so_far(plotting.load_results(plotting.ORDINARY_CSV))
    horizon = min(heat[-1].generation, conventional[-1].generation)
    heat = [c for c in heat if c.generation <= horizon]
    conventional = [c for c in conventional if c.generation <= horizon]
    output = plotting.BASE_DIR / 'wse2_total_loss_convergence_ideal_reference.png'
    plotting.plot_metric(heat, conventional, 'score', 'Loss',
                         (plotting.LOSS_Y_MIN, plotting.LOSS_Y_MAX), output,
                         ideal_reference=True)
    energy = plotting.load_energy_reference()
    output.with_suffix('.json').write_text(json.dumps({
        'reference_source': str(plotting.ENERGY_REFERENCE_CSV.relative_to(plotting.BASE_DIR.parents[1])),
        'formation_energy_per_atom_eV': energy,
        'assumed_weighted_image_penalty': 0.0,
        'assumed_weighted_composition_penalty': 0.0,
        'assumed_weighted_sample_box_penalty': 0.0,
        'entropy_contribution': 0.0,
        'ideal_total_loss': energy,
        'interpretation': 'Hypothetical zero-penalty reference, not an evaluated total loss or proven lower bound.',
        'generation_endpoint': horizon,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
