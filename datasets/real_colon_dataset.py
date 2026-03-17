from __future__ import absolute_import, division, print_function

import os
import csv
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class REALColonDataset(MonoDataset):
    """
    Dataset class for REAL-Colon, modelled on SCAREDRAWDataset.

    REAL-Colon structure on disk:
      <data_root>/
        SSS-VVV_frames/           e.g. 001-001_frames/
          SSS-VVV_1.jpg           frames named <seq_id>_<t>.jpg (1-indexed, no padding)
          SSS-VVV_2.jpg
          ...
        SSS-VVV_annotation/       bounding-box XML -- NOT used here
        video_info.csv            fps, num_frames, endoscope_brand, ...
        lesion_info.csv           polyp metadata -- NOT used here

    Key facts:
      - Frame filenames: <seq_id>_<t>.jpg  (e.g. 001-001_42.jpg)
        Source: official annotation XML: <filename>SSS-VVV_t.jpg</filename>
        Index t is 1-indexed with NO zero-padding.
      - Native resolution: 1240 x 1080 (width x height)
      - Endoscope: Olympus CF-HQ290L (Olympus sequences) or Fujifilm (other sequences)
        The CF-HQ290L is a conventional wide-angle scope with ~170 deg FOV and
        standard perspective/mild-fisheye optics. It is NOT an omnidirectional
        fisheye like the CF-HQ190L used in C3VDv2 -- no undistortion is needed.
      - No depth GT, no pose GT -- self-supervised training only.

    Intrinsics:
      REAL-Colon ships no calibration data. We use the normalised K from
      SCAREDDataset (fx_n=0.82, fy_n=1.02, cx_n=cy_n=0.5), which was derived
      from Olympus endoscopes with a comparable ~170 deg FOV used in the SCARED
      dataset and in the original AF-SfMLearner pre-training. This is the most
      principled approximation available and ensures consistency with the
      pre-trained weights. For Fujifilm sequences in REAL-Colon (similar FOV
      class), the same K is used as a best-available approximation.

      MonoDataset stores K normalised (values in [0,1]) and reconstructs
      pixel-unit K per scale in __getitem__ by multiplying K[0,:] by W//(2^s)
      and K[1,:] by H//(2^s).

    Split file format (MonoDataset convention):
        <folder> <frame_index> <side>
      e.g.:
        001-001 42 l
      side is always 'l'. frame_index is the integer matching the filename.
      Generate with scripts/generate_splits_realcolon.py.
    """

    def __init__(self, *args, **kwargs):
        super(REALColonDataset, self).__init__(*args, **kwargs)

        # Normalised intrinsics -- same format and values as SCAREDDataset.
        # AF-SfMLearner was pre-trained with this K on Olympus endoscopes;
        # using it here ensures the self-supervised loss operates in a
        # consistent coordinate frame with the pre-trained weights.
        self.K = np.array(
            [
                [0.82, 0, 0.5, 0],
                [0, 1.02, 0.5, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Monocular only.
        self.side_map = {"l": "l"}

    # ------------------------------------------------------------------
    # MonoDataset abstract method overrides
    # ------------------------------------------------------------------

    def check_depth(self):
        """REAL-Colon provides no ground-truth depth."""
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        """
        Load a single RGB frame as a PIL Image.

        Args:
            folder:      sequence ID string, e.g. '001-001'
            frame_index: integer frame index (1-indexed, matching filename)
            side:        ignored (monocular)
            do_flip:     if True, flip horizontally
        """
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_image_path(self, folder, frame_index, side):
        """
        Resolve the absolute path to one frame.

        Naming convention (from official annotation XML):
            <filename>SSS-VVV_t.jpg</filename>
        =>  <data_root>/<folder>_frames/<folder>_<frame_index>.jpg

        frame_index is 1-indexed with no zero-padding.
        """
        fname = "{}_{}{}".format(folder, frame_index, self.img_ext)
        return os.path.join(self.data_path, "{}_frames".format(folder), fname)

    def get_depth(self, folder, frame_index, side, do_flip):
        """Not implemented -- REAL-Colon has no GT depth."""
        raise NotImplementedError(
            "REALColonDataset does not provide ground-truth depth. "
            "check_depth() returns False so this should never be called."
        )


# ---------------------------------------------------------------------------
# Helper: read video_info.csv
# ---------------------------------------------------------------------------


def load_video_info(data_root):
    """
    Parse video_info.csv and return:
        { seq_id: {'num_frames': int, 'fps': float, 'endoscope_brand': str} }

    video_info.csv columns (official README):
        unique_video_name, age, sex, endoscope_brand, fps,
        num_frames, num_lesions, bbps
    """
    csv_path = os.path.join(data_root, "video_info.csv")
    info = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row["unique_video_name"].strip()
            info[seq_id] = {
                "num_frames": int(row["num_frames"]),
                "fps": float(row["fps"]),
                "endoscope_brand": row["endoscope_brand"].strip(),
            }
    return info
