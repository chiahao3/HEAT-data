Ordinary GA WSe2 structures and ptychographic image comparison

generation_NNNN_best_loss.xsf: best total-loss structure available through each generation, copied byte-for-byte from the run's xsf_members directory. best_by_generation.csv identifies the original winner and flags incomplete generations.

best_full_padded.xsf: original best structure including its original vacuum cell.
best_cropped_280x280.xsf: retain atoms inside the central 280 x 280 measured-pixel footprint, removing 20 measured pixels (3.05128 A) from every lateral edge, in addition to excluding the original lateral vacuum. The original per-side vacuum spacing is retained around the smaller material region. Z coordinates and padded z extent are unchanged.
best_cropped_300x300.xsf: retain atoms inside the central 300 x 300 measured-pixel footprint, removing 10 measured pixels (1.52564 A) from every lateral edge, in addition to excluding the original lateral vacuum. The original per-side vacuum spacing is retained around the smaller material region. Z coordinates and padded z extent are unchanged.
best_cropped_320x320.xsf: retain atoms within the full measured XY footprint, preserving the original cell. No new relaxation is performed on the cropped structures.

images/experimental_zsum_cropped_*.tif: exact sum of the original experimental TIFF crop over all 12 measured slices (raw input units).
images/reconstructed_zsum_cropped_*.tif: original best XSF rendered with the run's double-Gaussian XY / Gaussian Z model, summed over the same 12 slices and converted to input TIFF units.
images/*_normalized_zsum_cropped_*.tif: corresponding arrays in the run's normalized intensity units.
Individual PNGs preserve exactly 280x280, 300x300, or 320x320 pixels. All comparison and individual PNGs share one intensity scale. Comparison PDFs/PNGs show the two sources side by side.

Reconstruction images are cropped AFTER rendering the full best structure, preserving contributions from atoms just outside the displayed subregion. They are not a fresh render of the atom-deleted cropped XSF.
export_manifest.json records physical bounds, pixel ranges, atom counts, model settings, numerical comparisons, and source provenance.
