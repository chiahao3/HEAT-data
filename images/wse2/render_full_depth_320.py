"""Render the full measured region without changing the 280-pixel figure.

Run first with Blender --background --factory-startup --python THIS_FILE,
then with python3 THIS_FILE to add physical axes.
"""
from pathlib import Path
import importlib.util
import json

PAPER = Path(__file__).resolve().parents[2]
OUT = PAPER / 'images/wse2/figure_320x320'
SPAN = 54.0
CONFIG = {'camera': {
    'horizontal_span_A': SPAN,
    'vertical_span_A': SPAN * 1200 / 3200,
    'center_experimental_xyz_A': [48.82048 / 2, 48.82048 / 2, 6.0],
    'image_size': [3200, 1200],
    'direction': '+y',
}, 'final_xsf': 'xsfs/wse2/xsfs_wse2_EAT/best_cropped_320x320.xsf',
    'final_atoms': 1497, 'measured_lateral_size_A': 48.82048}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import bpy
    except ImportError:
        finish = load_module('finish_280', PAPER / 'images/wse2/finish_figure_280.py')
        finish.OUT = OUT
        finish.add_depth_axes('heat_final_depth', CONFIG)
        (OUT / 'figure_metadata.json').write_text(json.dumps(CONFIG, indent=2) + '\n')
        return
    style = load_module('structure_style', PAPER / 'scripts/blender/wse2/structure_style.py')
    style.INPUT_XSF = PAPER / CONFIG['final_xsf']
    style.OUTPUT_BLEND = OUT / 'heat_final_depth.blend'
    style.SIDE_OUTPUT_RENDER = OUT / 'heat_final_depth.png'
    style.VIEW_MODE = 'side'
    style.SIDE_IMAGE_SIZE = (3200, 1200)
    style.SIDE_CAMERA_ORTHO_SCALE = SPAN
    style.CROP_LATERAL_EDGE_ATOMS = False
    style.SHOW_UNIT_CELL = False
    style.NO_SAVE = False
    style.NO_RENDER = False
    bpy.context.preferences.filepaths.save_version = 0
    style.main()


if __name__ == '__main__':
    main()
