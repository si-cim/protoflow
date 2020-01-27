"""Pretty plot confusion matrix with seaborn.

Created on: Mon Jun 25 14:17:37 2018
Author: Wagner Cipriano - wagnerbhbr - gmail - CEFETMG / MMC
Modified on: Fri Nov 29 15:45:20 2019
Modified by: Jensun Ravichandran <jjohnrav@hs-mittweida.de>

REFerences:
  https://www.mathworks.com/help/nnet/ref/plotconfusion.html
  https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
  https://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python
  https://www.programcreek.com/python/example/96197/seaborn.heatmap
  https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/31720054
  http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
"""

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh


def get_new_fig(fn, figsize=[9, 9]):
    """Init graphics."""
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()  # get current axis
    ax1.cla()  # clear existing
    return fig1, ax1


def configcell_text_and_colors(array_df,
                               lin,
                               col,
                               oText,
                               facecolors,
                               posi,
                               fz,
                               fmt,
                               show_null_values=0):
    """Configure cell text and colors and return text
    elements to add and delete.

        TODO:
    """
    text_add = []
    text_del = []
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:, col]
    ccl = len(curr_column)

    # last line and/or last column
    if (col == (ccl - 1)) or (lin == (ccl - 1)):
        # tots and percents
        if (cell_val != 0):
            if (col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif (col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif (lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        # per_ok_s = ['%.2f%%' % (per_ok), '100%'][per_ok == 100]
        # DeprecationWarning: In future, it will be an error for
        # 'np.bool_' scalars to be interpreted as an index

        per_ok_s = '100%' if per_ok == 100 else f'{per_ok:05.02f}%'

        # text to DEL
        text_del.append(oText)

        # text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w',
                           ha='center',
                           va='center',
                           gid='sum',
                           fontproperties=font_prop)
        lis_txt = ['%d' % (cell_val), per_ok_s, '%.2f%%' % (per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy()
        dic['color'] = 'g'
        lis_kwa.append(dic)
        dic = text_kwargs.copy()
        dic['color'] = 'r'
        lis_kwa.append(dic)
        lis_pos = [(oText._x, oText._y - 0.3), (oText._x, oText._y),
                   (oText._x, oText._y + 0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0],
                           y=lis_pos[i][1],
                           text=lis_txt[i],
                           kw=lis_kwa[i])
            # print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        # print '\n'

        # Set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if (col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if (per > 0):
            txt = '%s\n%.2f%%' % (cell_val, per)
        else:
            if (show_null_values == 0):
                txt = ''
            elif (show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        # main diagonal
        if (col == lin):
            # Set color of the textin the diagonal to white.
            oText.set_color('w')
            # Set background color in the diagonal to blue.
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del


def insert_totals(df_cm):
    """Insert total column and line (the last ones)."""
    sum_col = []
    for c in df_cm.columns:
        sum_col.append(df_cm[c].sum())
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append(item_line[1].sum())
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col
    # print ('\ndf_cm:\n', df_cm, '\n\b\n')


def plot_cm(df_cm,
            annot=True,
            cmap='Oranges',
            fmt='.2f',
            fz=11,
            lw=0.5,
            cbar=False,
            figsize=[8, 8],
            show_null_values=0,
            pred_val_axis='y'):
    """Plot the confusion matrix with default layout (like matlab).

    params:
        df_cm          Dataframe (pandas) without totals.
        annot          Print text in each cell.
        cmap           Oranges, Oranges_r, YlGnBu, Blues, RdBu, ...
        fz             Fontsize.
        lw             Linewidth.
        pred_val_axis  Where to show the prediction values (x or y axis).
            'col' or 'x': show predicted values in columns (x axis)
                instead of lines.
            'lin' or 'y': show predicted values in lines (y axis).
    """
    try:
        import seaborn as sn
    except ModuleNotFoundError as e:
        print('Please install Protoflow with [other] extra requirements.')
        raise (e)

    if (pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create 'Total' column
    insert_totals(df_cm)

    # This is to always print in the same window.
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    # Thanks for seaborn
    ax = sn.heatmap(df_cm,
                    annot=annot,
                    annot_kws={'size': fz},
                    linewidths=lw,
                    ax=ax1,
                    cbar=cbar,
                    cmap=cmap,
                    linecolor='w',
                    fmt=fmt)

    # set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        # t.tick1On = False  # deprecated in Matplotlib 3.1
        # t.tick2On = False  # deprecated in Matplotlib 3.1
        t.tick1line.set_visible(False)
        t.tick2line.set_visible(False)
    for t in ax.yaxis.get_major_ticks():
        # t.tick1On = False  # deprecated in Matplotlib 3.1
        # t.tick2On = False  # deprecated in Matplotlib 3.1
        t.tick1line.set_visible(False)
        t.tick2line.set_visible(False)

    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist())
    text_add = []
    text_del = []
    posi = -1  # from left to right, bottom to top.
    for t in ax.collections[0].axes.texts:  # ax.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(pos[1])
        col = int(pos[0])
        posi += 1
        # print('>>> pos: %s, posi: %s, val: %s, txt: %s' %
        #       (pos, posi, array_df[lin][col], t.get_text()))

        # set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors,
                                             posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # Remove the old ones.
    for item in text_del:
        item.remove()
    # Append the new ones.
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    # Set titles and legends.
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    xlim = ax.get_xlim()
    ax.set_ylim(xlim[::-1])
    plt.tight_layout()  # set layout slim
    plt.show()


def confusion_matrix(y_test,
                     predictions,
                     columns=None,
                     annot=True,
                     cmap='Oranges',
                     fmt='.2f',
                     fz=11,
                     lw=0.5,
                     cbar=False,
                     figsize=[8, 8],
                     show_null_values=0,
                     pred_val_axis='lin',
                     return_numpy=False):
    """Compute and plot confusion matrix from `y_test` and `predictions`.

        TODO:
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    if (not columns):
        # columns = range(1, len(np.unique(y_test)) + 1)
        from string import ascii_uppercase
        columns = [
            'class %s' % (i)
            for i in list(ascii_uppercase)[0:len(np.unique(y_test))]
        ]

    confm = confusion_matrix(y_test, predictions)
    fz = 11
    figsize = [9, 9]
    show_null_values = 2
    df_cm = DataFrame(confm, index=columns, columns=columns)
    plot_cm(df_cm,
            fz=fz,
            cmap=cmap,
            figsize=figsize,
            show_null_values=show_null_values,
            pred_val_axis=pred_val_axis)
    if return_numpy:
        return df_cm.to_numpy()
    return df_cm
