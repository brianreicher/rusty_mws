import numpy as np
from scipy.ndimage import measurements
from funlib.segment.arrays import replace_values


def filter_fragments(affs: np.ndarray, fragments_data:np.ndarray, filter_val:float) -> np.ndarray:
    # try:
    average_affs: float = np.mean(affs.data, axis=0)

    filtered_fragments: list = []

    fragment_ids: np.ndarray = np.unique(fragments_data)

    for fragment, mean in zip(
        fragment_ids, measurements.mean(average_affs, fragments_data, fragment_ids)
    ):
        if mean < filter_val:
            filtered_fragments.append(fragment)

    filtered_fragments: np.ndarray = np.array(filtered_fragments, dtype=fragments_data.dtype)
    replace: np.ndarray = np.zeros_like(filtered_fragments)
    replace_values(fragments_data, filtered_fragments, replace, inplace=True)
    # except:
    #     pass
