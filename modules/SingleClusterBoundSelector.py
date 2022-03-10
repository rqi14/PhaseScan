import json
import os

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
# from scipy.optimize import curve_fit
# import matplotlib as mpl
# mpl.use('Qt5Agg')
from matplotlib.widgets import TextBox
from scipy.spatial import Delaunay
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN

from modules.proc_utils import find_slice_idx
from modules.utils import get_output_path


# https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points
def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def find_edges_with(i, edge_set):
    i_first = [j for (x, j) in edge_set if x == i]
    i_second = [j for (j, x) in edge_set if x == i]
    return i_first, i_second


def stitch_boundaries(edges):
    edge_set = edges.copy()
    boundary_lst = []
    while len(edge_set) > 0:
        boundary = []
        edge0 = edge_set.pop()
        boundary.append(edge0)
        last_edge = edge0
        while len(edge_set) > 0:
            i, j = last_edge
            j_first, j_second = find_edges_with(j, edge_set)
            if j_first:
                edge_set.remove((j, j_first[0]))
                edge_with_j = (j, j_first[0])
                boundary.append(edge_with_j)
                last_edge = edge_with_j
            elif j_second:
                edge_set.remove((j_second[0], j))
                edge_with_j = (j, j_second[0])  # flip edge rep
                boundary.append(edge_with_j)
                last_edge = edge_with_j

            if edge0[0] == last_edge[1]:
                break

        boundary_lst.append(boundary)
    return boundary_lst


# END  https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points

class SingleClusterBoundSelector:
    def __init__(self, df, col_names, params, group_feature_label=False):
        """
        Polynomial fitting of boundary of a sinlge cluster
        :param df: Dataframe without any incorrect data points
        :param col_names: list of two column names for x_data and y_data axes
        :param group_feature_label: which feature group to be used for polynomial fitting
        """
        if group_feature_label is None:
            self.df_oi = df.copy()
        else:
            self.df_oi = df[df.feature == group_feature_label]
        self.df_complete = df.copy()
        self.col_names = col_names.copy()
        self.fig_handle = None
        self.xy = None
        self.z = None
        self.vertices_idx = None
        self.edges = None
        self.fit_obj = None
        self._fit_handle = None
        self.fig_ax = None
        self.group_feature_label = group_feature_label
        self.params = params
        self.poly_const = None
        self.seq_idx = None

    def run(self):
        plt.ion()
        self.z = 'k'
        self.outlier_removal_dbscan()
        self.outlier_removal_density(25)
        xy = self.df_oi[self.col_names].to_numpy()
        self.xy = xy
        self.vertices_idx, self.edges = self.concave_hull()
        self.plot_scatter(self.z)
        self.plot_edges(self.edges)
        self.plot_the_other_cluster()
        self.launch_select_gui()

        # after selection
        self.plot_scatter('r', full=True)
        self.plot_the_other_cluster()
        x = self.xy[:, 0]
        x_range = np.linspace(x.min(), x.max(), 20)
        plt.plot(x_range, self.poly1d_obj(x_range), '--', color='#4b0082', linewidth=4)
        plt.ioff()
        self.fig_handle.savefig(os.path.join(get_output_path(self.params), 'single_fit.png'))

        # export result to json
        meta = {
            "polynomial_constants": self.poly_const.tolist(),
            "selected_sequence_indices": self.seq_idx.tolist()
        }
        with open(os.path.join(get_output_path(self.params), 'fitting_meta.json'), 'w+') as handle:
            json.dump(meta, handle)

    def outlier_removal_dbscan(self):
        clusters = DBSCAN(eps=0.3, min_samples=1).fit_predict(self.df_oi[self.col_names])
        (cluster_values, cluster_counts) = np.unique(clusters, return_counts=True)

        # plot for testing only
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(cluster_values))]
        xy = self.df_oi[self.col_names].to_numpy()
        for k, col in zip(cluster_values, colors):
            cluster_member_mask = (clusters == k)
            xyc = xy[cluster_member_mask]
            plt.scatter(xyc[:, 0], xyc[:, 1], c=col,
                        label=f'{k}')

        cluster_most = cluster_values[np.argmax(cluster_counts)]
        self.df_oi = self.df_oi.iloc[np.where(clusters == cluster_most)]

    def outlier_removal_density(self, percentage_threshold=25):
        # https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
        xy = self.df_oi[self.col_names].to_numpy()
        x = xy[:, 0]
        y = xy[:, 1]
        xy_h = xy.T  # two horizontal rows
        z = gaussian_kde(xy_h)(xy_h)
        idx2 = np.where(z > np.nanpercentile(z, percentage_threshold))[0]
        self.df_oi = self.df_oi.iloc[idx2]
        self.xy = xy[idx2]
        self.z = z[idx2]

    def plot_scatter(self, c='b', s=2, full=False):
        self.fig_handle = plt.figure(num="Select critical points")
        self.fig_ax = self.fig_handle.gca()
        if full:
            xy = self.df_complete[self.col_names].to_numpy()
        else:
            xy = self.xy
        plt.scatter(xy[:, 0], xy[:, 1], c=c, s=s, label=self.get_phase_display_label(self.group_feature_label))
        # plt.plot()
        plt.xlabel(f'{self.params.channels[0]} concentration')
        plt.ylabel(f'{self.params.channels[1]} concentration')
        plt.legend()

    def get_phase_display_label(self, feature):
        if feature is None:
            display_label = None
        else:
            display_label = 'Phase separation' if feature else 'Mixed'
        return display_label

    def plot_the_other_cluster(self, c='b', s=2):
        if self.group_feature_label is None:
            return
        xy_other = self.df_complete[self.df_complete.feature == (not self.group_feature_label)][
            self.col_names].to_numpy()
        plt.scatter(xy_other[:, 0], xy_other[:, 1], c=c, s=s,
                    label=self.get_phase_display_label(not self.group_feature_label))
        plt.legend()

    def plot_edges(self, edges):
        for i, j in edges:
            plt.plot(self.xy[[i, j], 0], self.xy[[i, j], 1], '-')
        for i in range(len(self.vertices_idx)):
            px, py = self.xy[self.vertices_idx[i]]
            plt.annotate(f'{i}', [px, py])

    def concave_hull(self):
        # Computing the alpha shape
        xy = self.df_oi[self.col_names].to_numpy()
        edges = alpha_shape(xy, alpha=5, only_outer=True)
        vertices_idx = np.array(stitch_boundaries(edges)[0])[:, 0]
        return vertices_idx, edges

    def launch_select_gui(self):
        print('Please input critical vertices in order separated by comma, and append the order of polynomial after a '
              'space.\n For example, to select 20,21,22,0,1,2,3,4,5 and fit a third order polynomial, simply input '
              '"20,0, 5 3" or "20,22,2,4,5 3" and press enter')

        def submit(seq_text):
            self.fig_handle.canvas.draw_idle()
            if self._fit_handle is not None:
                try:
                    self._fit_handle.remove()
                except ValueError:
                    print('Could not remove old curve')
            degree_split = seq_text.split(' ')
            poly_degree = int(degree_split[1])
            critical_point_idx = np.array(degree_split[0].split(','), np.int)
            seq_idx = find_slice_idx(np.array(range(len(self.vertices_idx))), critical_point_idx)
            vertices_selected = self.xy[np.array(self.vertices_idx)[seq_idx]]
            x_selected = vertices_selected[:, 0]
            y_selected = vertices_selected[:, 1]
            p = np.polyfit(x_selected, y_selected, poly_degree)
            x_range = np.linspace(x_selected.min(), x_selected.max(), 20)
            self.poly1d_obj = np.poly1d(p)
            print(f'Polynomial constants {p}')
            print(f'Selected indices {seq_idx}')
            self.poly_const = p
            self.seq_idx = seq_idx
            self._fit_handle, = self.fig_ax.plot(x_range, self.poly1d_obj(x_range), '--', color='#4b0082', linewidth=4)
            # plt.legend()
            self.fig_handle.canvas.draw_idle()
            # self.fig_ax.draw_idle()

        self.fig_handle.subplots_adjust(bottom=0.25)
        axbox = self.fig_handle.add_axes([0.2, 0.05, 0.7, 0.075])
        text_box = TextBox(axbox, 'Fit polynomial', initial='')
        text_box.on_submit(submit)
        plt.show(block=True)
