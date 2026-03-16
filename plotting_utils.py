import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

def plot_region_voronoi(
    cell_coords,
    type_vec,
    color_vec,
    cell_to_color_mapping,
    regions,
    vertices, 
    figsize=(8,10), 
    alpha=0.8, 
    edgecolor='black',
    prc_padding=0.05,
    save_fname=None):
    
    row_min, row_max = np.min(cell_coords[:,0]), np.max(cell_coords[:,0])
    col_min, col_max = np.min(cell_coords[:,1]), np.max(cell_coords[:,1])
    
    unique_cell_types = np.unique(type_vec)
    # convert from string to int then back
    # increment as well
    unique_cell_types = np.sort(unique_cell_types.astype(int))
    unique_cell_types = np.array([str(set_type) for set_type in unique_cell_types])
    
    fig, axes = plt.subplots(figsize=figsize)
    alpha=alpha
    for region_index, region in enumerate(regions):
        polygon = vertices[region]
        color_val = matplotlib.colors.to_rgba(color_vec[region_index], alpha)
        axes.fill(*zip(*polygon), alpha=alpha, color=color_val, edgecolor=edgecolor)
        
    plt.xlim(col_min-prc_padding, col_max+prc_padding)
    plt.ylim(row_min-prc_padding, row_max+prc_padding)
    
    custom_handles = []
    for cell_type in unique_cell_types:
        custom_handles.append(Line2D([], [], marker='.', color=cell_to_color_mapping[cell_type], linestyle='None'))

    plt.axis('off')
    plt.gca().invert_yaxis()

    # plt.legend(
    #     handles=custom_handles,
    #     labels=unique_cell_types.tolist(),
    #     bbox_to_anchor= (1.0, 1.0)
    # )
    
    if save_fname is not None:
        fig.savefig(save_fname, dpi=450, bbox_inches='tight')


def plot_wsi_scatter(cell_coords, type_vec, cell_to_color_mapping, sample_name, outcome, tissue_type, size=1, figure_size=(8,10), prc_padding=0.05, bar_len=0, flip_x=False, save_fname=None):
    
    row_min, row_max = np.min(cell_coords[:,1]), np.max(cell_coords[:,1])
    col_min, col_max = np.min(cell_coords[:,0]), np.max(cell_coords[:,0])
    
    unique_cell_types = np.unique(type_vec)
    # convert from string to int then back
    # increment as well
    if unique_cell_types.dtype!='object':
        unique_cell_types = np.sort(unique_cell_types.astype(int))
    unique_cell_types = np.array([str(set_type) for set_type in unique_cell_types])
    
    color_vec = np.vectorize(cell_to_color_mapping.get)(type_vec)
    print(color_vec.shape)
    fig, axes = plt.subplots(figsize=figure_size)
    
    plt.xlim(col_min-prc_padding, col_max+prc_padding)
    plt.ylim(row_min-prc_padding, row_max+prc_padding)
    
    # plt.axis('off')
    plt.gca().invert_yaxis()
    if flip_x:
        plt.gca().invert_xaxis()
    axes.set_title(f'{sample_name} - {outcome} - {tissue_type}')
    axes.scatter(cell_coords[:,0], cell_coords[:,1], s=size, color=color_vec, linewidths=0.2)
    
    if bar_len>0:
        if bar_len>1000:
            add_scalebar_um(axes, bar_len, label=f'{bar_len/1000:g} mm', loc='lower right', pad=0.8, font_size=12, color='black')
        else:
            add_scalebar_um(axes, bar_len, label=f'{bar_len} μm', loc='lower right', pad=0.8, font_size=12, color='black')

    
    custom_handles = []
    for cell_type in unique_cell_types:
        custom_handles.append(Line2D([], [], marker='.', color=cell_to_color_mapping[cell_type], linestyle='None'))
        

    plt.legend(
        handles=custom_handles,
        labels=unique_cell_types.tolist(),
        bbox_to_anchor= (1.0, 1.0)
    )
    
    if save_fname is not None:
        fig.savefig(save_fname, dpi=450, bbox_inches='tight')

    # plt.show()
    

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plot_neighbor_barstack(adata, neighbor_key, type_key, cell_color_dict, save_fname=None):
    unique_cluster_labels = np.unique(adata.obs[neighbor_key])
    # increment before going back
    unique_cluster_labels = np.sort(unique_cluster_labels.astype(int))
    unique_cluster_labels = np.array([str(neighbor_label) for neighbor_label in unique_cluster_labels])
    print(unique_cluster_labels)
    unique_cell_types = np.unique(adata.obs[type_key])
    neighbor_type_proportion_dict = {}
    
    for unique_label in unique_cluster_labels:
        sub_cell_proportion_dict = {cell_type: 0 for cell_type in unique_cell_types}
        neighbor_subset_frame = adata[adata.obs[neighbor_key]==unique_label]
        # get counts of cell types belonging to that label
        neighbor_cell_types, neighbor_type_counts = np.unique(neighbor_subset_frame.obs[type_key], return_counts=True)

        for present_cell_index in range(len(neighbor_cell_types)):
            sub_cell_proportion_dict[neighbor_cell_types[present_cell_index]] += neighbor_type_counts[present_cell_index]

        neighbor_type_proportion_dict[unique_label] = sub_cell_proportion_dict

    cell_neighbor_proportion_frame = pd.DataFrame(neighbor_type_proportion_dict).T
    
    # normalize the values in the dataframe between 0 and 1
    norm_term = np.sum(cell_neighbor_proportion_frame.values, axis=1).reshape(-1,1)
    norm_values = cell_neighbor_proportion_frame.values/norm_term
    norm_cell_neighbor_frame = pd.DataFrame(data=norm_values, index=cell_neighbor_proportion_frame.index, columns=cell_neighbor_proportion_frame.columns)
    
    neighborhood_name_mapping = {label: f'RCN {label}' for count, label in enumerate(unique_cluster_labels)}
    cell_types = norm_cell_neighbor_frame.columns.tolist()
    cell_colors = [cell_color_dict[cell_type] for cell_type in cell_types]

    fig, axes = plt.subplots(figsize=(8,4))
    # axes.set_xticklabels([f'{neighborhood_name_mapping[int(label)]}' for label in neighbor_frame.columns.tolist()], rotation=45, ha='right')
    axes.set_xticklabels([f'{neighborhood_name_mapping[label]}' for label in unique_cluster_labels], rotation=45, ha='right')
    bottom = np.zeros(len(unique_cluster_labels))
    neighborhood_ids = norm_cell_neighbor_frame.index.tolist()
    # make the ids categories
    neighborhood_ids = [str(neighbor_id) for neighbor_id in neighborhood_ids]
    cell_types = norm_cell_neighbor_frame.columns.tolist()
    width=0.8
    label_fmt = dict(size=len(unique_cluster_labels), labelpad=15)

    for count, cell_type in enumerate(cell_types):
        p = axes.bar(neighborhood_ids, norm_cell_neighbor_frame[cell_type].values, width, label=cell_type, color=cell_colors[count], edgecolor='black', linewidth=1, bottom=bottom)
        bottom+=norm_cell_neighbor_frame[cell_type].values

    axes.set_title(f'Cell Type Proportions within Each Neighborhood')
    axes.set_ylabel('Cell Type Proportions', **label_fmt)
    axes.set_xlabel('Neighborhood IDs', **label_fmt)

    axes.legend(bbox_to_anchor=(1.02, 1.00))
    if save_fname is not None:
        fig.savefig(save_fname, dpi=450, bbox_inches='tight')
    # fig.savefig(f'{barplot_dir}/{panel_name}_cell_threshold_{region_cell_threshold}_sample_{prc_sample}_clusters_{num_clusters}.svg', dpi=450, bbox_inches='tight')
    plt.show()
    
    return norm_cell_neighbor_frame
    
def plot_stacked_bar(composition_frame, color_mapping, title, xlabel, ylabel, save_fname=None, linewidth=1, drop_x=True, edgecolor='black', width=0.8, figsize=(5,5)):
    # makes a stacked barplot where columns from a datafram are the stacks
    # and dataframe rows are the x vals
    composition_rows = composition_frame.index.tolist()
    composition_cols = composition_frame.columns.tolist()
    fig, axes = plt.subplots(figsize=figsize)
    # axes.set_xticklabels([f'{neighborhood_name_mapping[int(label)]}' for label in neighbor_frame.columns.tolist()], rotation=45, ha='right')
    if not drop_x:
        axes.set_xticklabels([row_name for row_name in composition_rows])
    else:
        axes.set_xticklabels([])

    bottom = np.zeros(len(composition_rows))
    width=0.8
    label_fmt = dict(size=len(composition_cols), labelpad=15)

    for count, col_type in enumerate(composition_cols):
        p = axes.bar(composition_rows, composition_frame[col_type].values, width, label=col_type, color=color_mapping[col_type], bottom=bottom, edgecolor='black', linewidth=1)
        bottom+=composition_frame[col_type].values

    axes.set_title(title)
    axes.set_ylabel(ylabel, **label_fmt)
    axes.set_xlabel(xlabel, **label_fmt)

    axes.legend(bbox_to_anchor=(1.02, 1.00))
    if save_fname is not None:
        fig.savefig(save_fname, dpi=450, bbox_inches='tight')

def plot_stacked_bar_with_annotations(composition_frame, color_mapping, annotations, annotation_color_mapping,
                                      title, xlabel, ylabel, save_fname=None, linewidth=1, edgecolor='black',
                                      bar_width=0.2, figsize=(10, 6)):
    """
    composition_frame: DataFrame (samples x components)
    color_mapping: dict mapping components to colors
    annotations: dict mapping annotation name to list of values (length = number of samples)
    annotation_color_mapping: dict mapping annotation name to value:color dict
    """
    num_samples = composition_frame.shape[0]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=(len(annotations)+1), ncols=1, height_ratios=[0.2]*len(annotations) + [1.0])

    # Plot annotation bars
    for idx, (ann_name, ann_values) in enumerate(annotations.items()):
        ax_ann = fig.add_subplot(gs[idx, 0])
        for i, val in enumerate(ann_values):
            ax_ann.bar(i, 1, color=annotation_color_mapping[ann_name].get(val, 'white'), edgecolor='none', width=1)
        ax_ann.set_xlim(-0.5, num_samples - 0.5)
        ax_ann.set_xticks([])
        ax_ann.set_yticks([])
        ax_ann.set_ylabel(ann_name, rotation=0, ha='right', va='center')

    # Plot stacked bars
    ax = fig.add_subplot(gs[-1, 0])
    bottom = np.zeros(num_samples)
    x = np.arange(num_samples)

    for col in composition_frame.columns:
        ax.bar(x, composition_frame[col].values, bar_width,
               label=col, color=color_mapping[col], bottom=bottom,
               edgecolor=edgecolor, linewidth=linewidth)
        bottom += composition_frame[col].values

    ax.set_xlim(-0.5, num_samples - 0.5)
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')

    plt.tight_layout()
    if save_fname:
        plt.savefig(save_fname, dpi=450, bbox_inches='tight')
    plt.show()


def get_distinct_colors(n):
    """Generates `n` visually distinct colors in HEX format."""
    if n <= 12:
        # Use Tableau or Set3 palette for small numbers
        palette = sns.color_palette("Set3", n)
    elif n <= 20:
        palette = sns.color_palette("tab20", n)
    else:
        # Fall back to HSV if more colors are needed
        palette = [plt.cm.hsv(i / n) for i in range(n)]
    
    return [mcolors.to_hex(color) for color in palette]

def generate_neighborhood_color_map(min_neighbors=5, max_neighbors=20):
    """
    Generates a dict of {neighborhood_id: color} for each number of neighbors from min to max.
    Each count of neighbors gets its own color map.
    """
    all_maps = {}
    for k in range(min_neighbors, max_neighbors + 1):
        color_list = get_distinct_colors(k)
        id_to_color = {str(i): color_list[i] for i in range(k)}
        all_maps[k] = id_to_color
    return all_maps
   

def add_scalebar_um(ax, length_um, label=None, loc="lower right",
                    pad=0.4, borderpad=0.4, sep=4,
                    color="black", size_vertical_um=None, font_size=8):
    """
    Add a scale bar to an axes where data units are microns.

    length_um: bar length in microns (data units)
    label: text label (defaults to '<length> µm')
    size_vertical_um: bar thickness in microns (data units). If None, set to ~2% of length.
    """
    if label is None:
        label = f"{length_um:g} µm"
    if size_vertical_um is None:
        size_vertical_um = max(1.0, 0.02 * length_um)  # thickness in µm

    fontprops = fm.FontProperties(size=font_size)

    sb = AnchoredSizeBar(
        transform=ax.transData,
        size=length_um,
        label=label,
        loc=loc,
        pad=pad,
        borderpad=borderpad,
        sep=sep,
        color=color,
        size_vertical=size_vertical_um,
        frameon=True,
        fontproperties=fontprops
    )
    
    sb.patch.set_facecolor('white')
    sb.patch.set_edgecolor('none')
    sb.patch.set_alpha(0.8)
    ax.add_artist(sb)

    