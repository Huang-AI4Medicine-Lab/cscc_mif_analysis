import numpy as np
import pandas as pd
from skimage import filters
from scipy.spatial import cKDTree, distance
from tqdm import tqdm
import re


def coords_to_geojson(coord_dict):
    # feature_dict = {
    #     "type": "FeatureCollection",
    #     "features": []
    # }
    feature_list = []

    for idx, coord_key in enumerate(list(coord_dict.keys())):
        coord_sample = coord_dict[coord_key]
        top_left, bottom_left, bottom_right, top_right = coord_sample
        polygon_coordinates = [[top_left, top_right, bottom_right, bottom_left, top_left]]

        feature = {
            'type': 'Feature',
            "id": "PathROIObject",
            'geometry': {
                'type': 'Polygon',
                'coordinates': polygon_coordinates
            },
            "properties": {
                'isLocked': "false",
                'measurements': [],
                "classification": {"name": f"Region: {coord_key}", "colorRGB": -377282}
            }
        }

        # feature_dict['features'].append(feature)
        feature_list.append(feature)

    return feature_list

def get_row_col(filename):
    pattern = r"row_(\d+)_col_(\d+)"

    match = re.search(pattern, filename)

    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        # print(f"Row: {row}, Column: {col}")
    else:
        print("No match found.")
        
    return row, col


def average_distance_between_cell_types(df, type_a, type_b, ref_type, type_key='registered_cell_types', region_key='merged_region_name', x_key='registered_wsi_cell_centroid_x_um', y_key='registered_wsi_cell_centroid_y_um'):
    """
    Calculate the average minimum distance from each cell of type_a to the nearest cell of type_b, within each region.

    Parameters:
    - df: DataFrame with columns ['cell_type', 'x', 'y', 'region']
    - type_a: str, cell type from which distances are measured
    - type_b: str, cell type to which distances are measured

    Returns:
    - average_min_distance: float, the mean of the minimum distances from type_a to type_b across all regions
    """
    min_distances = []
    stored_region_names = []

    for region in tqdm(df[region_key].unique()):
        region_df = df[df[region_key] == region]
        coords_a = region_df[region_df[type_key] == type_a][[x_key, y_key]].to_numpy()
        coords_b = region_df[region_df[type_key] == type_b][[x_key, y_key]].to_numpy()
        
        coords_ref = region_df[region_df[type_key] == ref_type][[x_key, y_key]].to_numpy()

        if len(coords_a) == 0 or len(coords_b) == 0:
            continue
            
        if len(coords_ref)==0:
            continue

        dists = distance.cdist(coords_a, coords_b)
        region_min_distances = np.min(dists, axis=1)
        min_distances.extend(region_min_distances)
        stored_region_names.extend([region]*len(region_min_distances))
        # min_distances.append(np.mean(np.min(dists, axis=1)))

    if not min_distances:
        return np.nan  # No valid distances could be computed

    region_distance_frame = pd.DataFrame()
    region_distance_frame[region_key] = stored_region_names
    region_distance_frame[f'{type_a}_{type_b}_min_distances'] = np.array(min_distances)
    
    return np.mean(min_distances), np.array(min_distances), region_distance_frame



def feature_filter_regions(cell_frame, threshold_limit, region_key, nbins=50, save_dir=None):
    unique_regions, region_cell_counts = np.unique(cell_frame[region_key], return_counts=True)

    num_regions_past_threshold = np.sum(region_cell_counts>threshold_limit)

    print(f'Number of regions passing threshold: {num_regions_past_threshold}')

    if save_dir is not None:
        fig, axes = plt.subplots()
        axes.hist(region_cell_counts, bins=nbins)
        plt.axvline(threshold_limit, c='red')
        fig.savefig(os.path.join(save_dir, 'region_sampling_threshold_hist.jpg'), dpi=450)

    region_indices = np.nonzero(region_cell_counts>threshold_limit)[0]
    valid_regions = unique_regions[region_indices]

    return valid_regions


def remove_duplicate_centroids(centroids, radius=10, cell_types=None, consider_cell_type=False):
    """
    Remove duplicate centroids that are within a certain radius of each other.
    Optionally, only remove duplicates if they share the same cell type.

    Parameters:
    - centroids (np.ndarray): N x 2 array of (x, y) cell centroids.
    - radius (float): Radius within which centroids are considered duplicates.
    - cell_types (list or np.ndarray): N-length array of cell types matching centroids.
    - consider_cell_type (bool): If True, only remove duplicates that share the same cell type.

    Returns:
    - np.ndarray: Filtered centroids with duplicates removed.
    - list or np.ndarray: Corresponding filtered cell types (if provided), else None.
    """
    centroids = np.asarray(centroids)
    tree = cKDTree(centroids)

    to_remove = set()
    kept = set()

    for i, point in enumerate(centroids):
        if i in to_remove:
            continue
        indices = tree.query_ball_point(point, r=radius)
        for j in indices:
            if j == i or j in kept:
                continue
            if consider_cell_type:
                if cell_types is not None and cell_types[i] == cell_types[j]:
                    to_remove.add(j)
            else:
                to_remove.add(j)
        kept.add(i)

    mask = np.array([i not in to_remove for i in range(len(centroids))])
    return mask


def extract_tissue_mask(
        src_im, 
        reduce_method='max', 
        num_thresholds=1, 
        num_tiles=300,
        tile_width=100,
        thresh_mode='min'
    ):
    # compress the channels if image has more than 3 dims
    if src_im.ndim>2:
        if reduce_method=='max':
            reduced_im = src_im.max(axis=0)
        elif reduce_method=='mean':
            reduced_im = src_im.mean(axis=0)
    else:
        reduced_im = src_im

    sample_rows = np.random.choice(reduced_im.shape[0], num_tiles, replace=False)
    sample_cols = np.random.choice(reduced_im.shape[1], num_tiles, replace=False)

    intensity_tile_list = []
    for tile_index in range(num_tiles):
        stitch_im = reduced_im[sample_rows[tile_index]:sample_rows[tile_index]+tile_width, sample_cols[tile_index]:sample_cols[tile_index]+tile_width]
        if stitch_im.shape[0]!=tile_width or stitch_im.shape[1]!=tile_width:
            continue
        intensity_tile_list.append(stitch_im)

    stitched_intensity_im = np.concatenate(intensity_tile_list, axis=0)

    if num_thresholds>1:
        otsu_thresh = filters.threshold_multiotsu(stitched_intensity_im, classes=num_thresholds+1)
        if thresh_mode=='min':
            min_thresh = np.min(otsu_thresh)
        elif thresh_mode=='max':
            min_thresh = np.max(otsu_thresh)
    else:
        min_thresh = filters.threshold_otsu(stitched_intensity_im)

    bin_reduced_im = (reduced_im>min_thresh)

    return bin_reduced_im


# def remove_duplicate_centroids(centroids, radius=10, cell_types=None, consider_cell_type=False):
#     """
#     Remove duplicate centroids that are within a certain radius of each other.
#     Optionally, only remove duplicates if they share the same cell type.

#     Parameters:
#     - centroids (np.ndarray): N x 2 array of (x, y) cell centroids.
#     - radius (float): Radius within which centroids are considered duplicates.
#     - cell_types (list or np.ndarray): N-length array of cell types matching centroids.
#     - consider_cell_type (bool): If True, only remove duplicates that share the same cell type.

#     Returns:
#     - np.ndarray: Filtered centroids with duplicates removed.
#     - list or np.ndarray: Corresponding filtered cell types (if provided), else None.
#     """
#     centroids = np.asarray(centroids)
#     tree = cKDTree(centroids)

#     to_remove = set()
#     kept = set()

#     for i, point in enumerate(centroids):
#         if i in to_remove:
#             continue
#         indices = tree.query_ball_point(point, r=radius)
#         for j in indices:
#             if j == i or j in kept:
#                 continue
#             if consider_cell_type:
#                 if cell_types is not None and cell_types[i] == cell_types[j]:
#                     to_remove.add(j)
#             else:
#                 to_remove.add(j)
#         kept.add(i)

#     mask = np.array([i not in to_remove for i in range(len(centroids))])
#     return mask

