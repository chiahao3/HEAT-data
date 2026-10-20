Generation-128 WSe2 figure assets

Use these PNGs in the assembled figure:
  heat_generation_0128_reconstructed_zsum_280x280_axes.png
  ordinary_generation_0128_reconstructed_zsum_280x280_axes.png
  heat_generation_0128_depth.png

Projection SVG/PDF variants are also supplied, together with unadorned PNGs
and numerical TIFFs. The depth view also has optional axes PNG/PDF variants.
All projections retain the existing 280x280 crop, common intensity scale,
24 pt axis labels, and 18.67 pt tick labels. They use the full structures before
cropping and the same 12 measured depth slices as experiment.

HEAT's best structure at generation 128 was first found at generation 115,
member 31. Its depth image is reused after checking identical coordinates and
species. Ordinary uses generation 128, member 33. Both weighted image losses
are checked against the run metrics. See render_metadata.json for provenance.

The experimental and generation-0 panels remain valid. Label the final
structure and both reconstruction panels as Generation 128 in the SVG.

Regenerate from the paper directory:
  python3 images/wse2/render_generation128.py
