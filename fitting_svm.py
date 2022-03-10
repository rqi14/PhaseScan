import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from modules.ParamLoading import ParamLoader
from modules.proc_utils import plot_detection_result
from modules.utils import get_output_path, get_output_df_path, argv_proc, read_effective_df
from typing import List
from skimage import measure

import uuid


def main(argv=None):
    ag = argv_proc(argv, sys.argv)
    params = ParamLoader(ag[0])
    svm_fit(params)

def svm_fit(params):
    df = read_effective_df(get_output_df_path(params))
    svm_fit_core(df, params)

def svm_fit_core(df, params):


    feature_col_names = [f"signal_{i}_pv{'_cvt' if params.train_on_cvt else ''}" for i in params.ch_train_idx]
    df_features_class = df[[*feature_col_names, 'feature']]

    # Iterate columns, normalise
    df_features_norm = df_features_class.copy()
    l_mean = []
    l_max = []
    l_min = []
    for column_name, column_data in df_features_norm.iteritems():
        if column_name == 'feature':
            continue
        col_mean = np.mean(column_data)
        col_max = np.max(column_data)
        col_min = np.min(column_data)
        col_range = col_max - col_min if col_max - col_min > 0 else 1
        df_features_norm[column_name] = (column_data - col_mean) / col_range
        # df_features_norm[column_name] = column_data
        l_mean.append(col_mean)
        l_max.append(col_max)
        l_min.append(col_min)
    df_features_norm_shuffled = df_features_norm.sample(frac=1)
    df_features = df_features_norm_shuffled[feature_col_names]
    df_class = df_features_norm_shuffled['feature']

    # Set training and testing data
    train_size = int(df_features.shape[0] * 1)
    X_train = df_features.values[0:train_size]
    y_train = df_class.values[0:train_size]

    #    X_test = df_features.values[train_size::]
    #    y_test = df_class.values[train_size::]
    X_test = X_train.copy()
    y_test = y_train.copy()
    # X_train, X_test, y_train, y_test = train_test_split(
    #    X, y_data, test_size=0.5, random_state=0)

    # see https://blog.csdn.net/aliceyangxi1987/article/details/73769950 for tutorial of grid search
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                         'C': [1, 10, 100, 1000, 10000, 100000]}]
    scores = ['f1']  # type of scores

    for score in scores:
        print(f"# Tuning hyper-parameters for {score} with grid search")
        print()
        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        if y_train.min() == y_train.max():
            print('There is only one class of data. Boundary cannot be fitted. The analysis is terminated')
            return
        clf.fit(X_train, y_train)

        print(f"Best parameters found:\n {clf.best_params_}")
        print("Grid search scores:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']

        for mean, std, clf_params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, clf_params))

        y_true, y_pred = y_test, clf.predict(X_test)

        print(classification_report(y_true, y_pred))

        # Calculate boundary
        train_size = int(df_features.shape[0] * 1)
        features_train = df_features.values[0:train_size]
        class_train = df_class.values[0:train_size]

        # features_test = df_features.values[train_size::]
        # class_test = df_class.values[train_size::]

        classifier = svm.SVC(**clf.best_params_)
        classifier.fit(features_train, class_train)

        # Training complete, scanning the space to extract result

        X_set, y_set = features_train, class_train

        # X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min(), stop=X_set[:, 0].max(), step=(X_set[:, 0].max() - X_set[:, 0].min()) / 500),
        #                      np.arange(start=X_set[:, 1].min(), stop=X_set[:, 1].max(),
        #                                step=(X_set[:, 1].max() - X_set[:, 1].min()) / 500))  # type: Tuple[np.ndarray]

        mesh_coords = [np.arange(start=X_set[:, i].min(), stop=X_set[:, i].max(),
                                   step=(X_set[:, i].max() - X_set[:, i].min()) / 100) for i in
                         range(len(params.ch_train_idx))]

        Xn = np.meshgrid(*mesh_coords)  # type: List[np.ndarray]
        # Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
        Z = classifier.predict(np.array([Xi.ravel() for Xi in Xn]).T).reshape(Xn[0].shape)

        if len(params.ch_plot_idx) == 2:
            plot_svm_2d(mesh_coords, Z, l_max, l_min, l_mean, X_set, y_set, params, df,feature_col_names)

        elif len(params.ch_plot_idx) == 3:
            plot_svm_3d(mesh_coords, Z, l_max, l_min, l_mean, params, df,feature_col_names)


def plot_svm_3d(mesh_coords, Z, l_max, l_min, l_mean, params, df, col_names, fn_override=None):
    # Xn_abs = [Xn[i] * (l_max[i] - l_min[i]) + l_mean[i] for i in range(len(Xn))]  # map Xn to original value
    mesh_coords_abs = [mesh_coords[i] * (l_max[i] - l_min[i]) + l_mean[i] for i in
                       range(len(mesh_coords))]  # map Xn to original value
    if Z.min() == True or Z.max() == False:
        print("Warning: Singlet data classification type")
        print("Stop plotting...")
        return
    verts, faces, normals, values = measure.marching_cubes(Z, 0, spacing=(1, 1, 1))
    verts[:, [1, 0]] = verts[:, [0, 1]]
    row_idx = np.array([[i for i in range(verts.shape[1])] for j in range(verts.shape[0])])
    verts_abs = np.array(mesh_coords_abs)[row_idx, verts.astype(np.int)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(20, -45)

    df_true = df[df.feature == True]
    xy_true = df_true[col_names].to_numpy()
    df_false = df[df.feature == False]
    xy_false = df_false[col_names].to_numpy()
    xt, yt, zt = xy_true.T
    xf, yf, zf = xy_false.T
    ax.scatter(xt, yt, zt, s=2)
    ax.scatter(xf, yf, zf, s=2)
    boundary_surf = ax.plot_trisurf(verts_abs[:, 0], verts_abs[:, 1], faces, verts_abs[:, 2],
                                    lw=1, color='orange', alpha=0.4)
    try:
        boundary_surf._facecolors2d = boundary_surf._facecolors3d
        boundary_surf._edgecolors2d = boundary_surf._edgecolors3d
    except:
        boundary_surf._facecolors2d = boundary_surf._facecolor3d
        boundary_surf._edgecolors2d = boundary_surf._edgecolor3d
    leg = plt.legend(['Phase separation', 'Mixed', 'Boundary'], bbox_to_anchor=(0.5, 1.05),
                     ncol=3, fancybox=True, shadow=True, loc='upper center')
    leg.legendHandles[0].set_color('blue')
    leg.legendHandles[1].set_color('red')
    leg.legendHandles[2].set_color('orange')
    # plt.show()
    plot_chs = params.channels[params.ch_plot_idx]
    if params.train_on_cvt == True:
        axis_labels = [f'{ch} concentration' for ch in plot_chs]
    else:
        axis_labels = [f'{ch} intensity per pixel volume' for ch in plot_chs]
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    fig.savefig(os.path.join(get_output_path(params), r'svm_3D.png'))

    # Plot density
    density_fig = plt.figure()
    ax = density_fig.add_subplot(111, projection='3d')
    sct = ax.scatter(xt, yt, zt, cmap=plt.get_cmap('Reds'), c=find_density(xy_true), s=2)
    scf = ax.scatter(xf, yf, zf, cmap=plt.get_cmap('Blues'), c=find_density(xy_false), s=2)
    boundary_surf = ax.plot_trisurf(verts_abs[:, 0], verts_abs[:, 1], faces, verts_abs[:, 2],
                                    lw=1, alpha=0.4, color='orange')
    try:
        boundary_surf._facecolors2d = boundary_surf._facecolors3d
        boundary_surf._edgecolors2d = boundary_surf._edgecolors3d
    except:
        boundary_surf._facecolors2d = boundary_surf._facecolor3d
        boundary_surf._edgecolors2d = boundary_surf._edgecolor3d
    leg = plt.legend(['Phase separation', 'Mixed', 'Boundary'], bbox_to_anchor=(0.7, 1.2),
                     ncol=3, fancybox=True, shadow=True, loc='upper center')
    leg.legendHandles[0].set_color('red')
    leg.legendHandles[1].set_color('blue')
    leg.legendHandles[2].set_color('orange')
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    density_fig.colorbar(sct, shrink=0.5, aspect=20, pad=0, format=fmt)
    density_fig.colorbar(scf, shrink=0.5, aspect=20, pad=0.15, format=fmt)

    ax.set_xlabel(axis_labels[0], fontsize=6)
    ax.set_ylabel(axis_labels[1], fontsize=6)
    ax.set_zlabel(axis_labels[2], fontsize=6)

    plt.show(block=False)

    if fn_override is None:
        fn_override = {}

    density_fig.savefig(os.path.join(get_output_path(params), fn_override.get(r'svm_density_3D.png', r'svm_density_3D.png')))

    # Save boundary surface
    np.savetxt(os.path.join(get_output_path(params), fn_override.get('boundary_svm_3d_vertices_abs.txt','boundary_svm_3d_vertices_abs.txt')), verts_abs)
    np.savetxt(os.path.join(get_output_path(params), fn_override.get('boundary_svm_3d_faces.txt','boundary_svm_3d_faces.txt')), faces)


def plot_svm_2d(mesh_coords, Z, l_max, l_min, l_mean, X_set, y_set, params, df, col_names, fn_override=None):
    if Z.min() == True or Z.max() == False:
        print("Warning: Singlet data classification type")
        print("Stop plotting...")
        return
    Xn = np.meshgrid(*mesh_coords)  # type: List[np.ndarray]
    q = Z.astype(np.uint8) ^ np.roll(Z.astype(np.uint8), shift=-1)
    boundary_y_b, boundary_x_b = np.where(q[:, 0:q.shape[1] - 1])  # This is the indices of boundary
    boundary_x = mesh_coords[0][boundary_x_b]  # This is the abs value (norm) of boundary
    boundary_y = mesh_coords[1][boundary_y_b]  # This is the abs value (norm) of boundary

    boundary_x = boundary_x * (l_max[0] - l_min[0]) + l_mean[0]  # map to original value
    boundary_y = boundary_y * (l_max[1] - l_min[1]) + l_mean[1]  # map to original value

    # X1_abs = X * (l_max[0] - l_min[0]) + l_mean[0]
    # X2_abs = X2 * (l_max[1] - l_min[1]) + l_mean[1]
    Xn_abs = [Xn[i] * (l_max[i] - l_min[i]) + l_mean[i] for i in range(len(Xn))]  # map Xn to original value

    # plot contourf
    cf_hd = plt.figure()
    plt.contourf(Xn_abs[0], Xn_abs[1], Z,
                 alpha=0.75, cmap=ListedColormap(('orange', 'green')))
    plt.xlim(Xn_abs[0].min(), Xn_abs[0].max())
    plt.ylim(Xn_abs[1].min(), Xn_abs[1].max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0] * (l_max[0] - l_min[0]) + l_mean[0],
                    X_set[y_set == j, 1] * (l_max[1] - l_min[1]) + l_mean[1],
                    color=ListedColormap(('red', 'green'))(i), label=j, marker='.')
    plt.xlabel(params.channels[params.ch_plot_idx[0]])
    plt.ylabel(params.channels[params.ch_plot_idx[1]])
    plt.legend()
    plt.show(block=False)

    handle = plot_detection_result(
        df, x_name=col_names[0], y_name=col_names[1], annotate=False,
        window_name=f'detection result {str(uuid.uuid4())}')
    # bp = np.sort(np.vstack([boundary_x, boundary_y]).T)
    plt.scatter(boundary_x, boundary_y, s=2)
    # plt.plot(boundary_x, boundary_y)
    # plt.legend([''])
    plt.show(block=False)
    cf_hd.savefig(os.path.join(get_output_path(params), r'svm_contour_filled.png'))

    # Save boundary values
    bd_out = np.array([boundary_x, boundary_y]).T
    if fn_override is None:
        fn_override = {}
    np.savetxt(os.path.join(get_output_path(params), 'boundary_svm.txt'), bd_out)

    density_handle = plt.figure(num='Density plot ' + str(uuid.uuid4()))
    df_true = df[df.feature == True]
    xy_true = df_true[col_names].to_numpy()
    df_false = df[df.feature == False]
    xy_false = df_false[col_names].to_numpy()
    plt.scatter(xy_true[:, 0], xy_true[:, 1], cmap=plt.get_cmap('Reds'), c=find_density(xy_true), s=2)
    plt.colorbar()
    plt.scatter(xy_false[:, 0], xy_false[:, 1], cmap=plt.get_cmap('Blues'), c=find_density(xy_false), s=2)
    plt.colorbar()
    plt.scatter(boundary_x, boundary_y, s=2, c='k')
    plt.legend(['Phase separation', 'Mixed', 'Boundary'])
    leg = plt.gca().get_legend()
    leg.legendHandles[0].set_color('red')
    leg.legendHandles[1].set_color('blue')
    leg.legendHandles[2].set_color('k')
    plt.xlabel(f'{params.channels[0]} Concentration')
    plt.ylabel(f'{params.channels[1]} Concentration')
    plt.show(block=False)
    density_handle.savefig(os.path.join(get_output_path(params), r'svm_density_2D.png'))


# Extra plot with density
def find_density(xy):
    xy_h = xy.T  # two horizontal rows
    return gaussian_kde(xy_h)(xy_h)

if __name__ == "__main__":
    main()
