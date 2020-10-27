import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

from protoflow.utils.celluloid import Camera
from protoflow.utils.colors import color_scheme
from protoflow.utils.utils import (gif_from_dir, make_directory,
                                   prettify_string, writelog)


class VisWeights(tf.keras.callbacks.Callback):
    """Abstract weight visualization callback."""
    def __init__(
            self,
            data=None,
            ignore_last_output_row=False,
            label_map=None,
            project_mesh=False,
            project_protos=False,
            voronoi=False,
            axis_off=True,
            cmap='viridis',
            show=True,
            display_logs=True,
            display_logs_settings={},
            pause_time=0.5,
            border=1,
            resolution=10,
            interval=False,
            save=False,
            snap=True,
            save_dir='./img',
            make_gif=False,
            make_mp4=False,
            verbose=True,
            dpi=500,
            fps=5,
            figsize=(11, 8.5),  # standard paper in inches
            prefix='',
            distance_layer_index=-1,
            **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.ignore_last_output_row = ignore_last_output_row
        self.label_map = label_map
        self.voronoi = voronoi
        self.axis_off = True
        self.project_mesh = project_mesh
        self.project_protos = project_protos
        self.cmap = cmap
        self.show = show
        self.display_logs = display_logs
        self.display_logs_settings = display_logs_settings
        self.pause_time = pause_time
        self.border = border
        self.resolution = resolution
        self.interval = interval
        self.save = save
        self.snap = snap
        self.save_dir = save_dir
        self.make_gif = make_gif
        self.make_mp4 = make_mp4
        self.verbose = verbose
        self.dpi = dpi
        self.fps = fps
        self.figsize = figsize
        self.prefix = prefix
        self.distance_layer_index = distance_layer_index
        self.title = 'Weights Visualization'
        make_directory(self.save_dir)

    def _skip_epoch(self, epoch):
        if self.interval:
            if epoch % self.interval != 0:
                return True
        return False

    def _clean_and_setup_ax(self):
        ax = self.ax
        if not self.snap:
            ax.cla()
        ax.set_title(self.title)
        if self.axis_off:
            ax.axis('off')

    def _savefig(self, fignum, orientation='horizontal'):
        figname = f'{self.save_dir}/{self.prefix}{fignum:05d}.png'
        figsize = self.figsize
        if orientation == 'vertical':
            figsize = figsize[::-1]
        elif orientation == 'horizontal':
            pass
        else:
            pass
        self.fig.set_size_inches(figsize, forward=False)
        self.fig.savefig(figname, dpi=self.dpi)

    def _show_and_save(self, epoch):
        if self.show:
            plt.pause(self.pause_time)
        if self.save:
            self._savefig(epoch)
        if self.snap:
            self.camera.snap()

    def _display_logs(self, ax, epoch, logs):
        if self.display_logs:
            settings = dict(
                loc='lower right',
                # padding between the text and bounding box
                pad=0.5,
                # padding between the bounding box and the axes
                borderpad=1.0,
                # https://matplotlib.org/api/text_api.html#matplotlib.text.Text
                prop=dict(
                    fontfamily='monospace',
                    fontweight='medium',
                    fontsize=12,
                ))

            # Override settings with self.display_logs_settings.
            settings = {**settings, **self.display_logs_settings}

            log_string = f"""Epoch: {epoch:04d},
            val_loss: {logs.get('val_loss', np.nan):.03f},
            val_acc: {logs.get('val_acc', np.nan):.03f},
            loss: {logs.get('loss', np.nan):.03f},
            acc: {logs.get('acc', np.nan):.03f}
            """
            log_string = prettify_string(log_string, end='')
            # https://matplotlib.org/api/offsetbox_api.html#matplotlib.offsetbox.AnchoredText
            anchored_text = AnchoredText(log_string, **settings)
            self.ax.add_artist(anchored_text)

    def on_train_begin(self, logs={}):
        self.fig = plt.figure(self.title)
        self.fig.set_size_inches(self.figsize, forward=False)
        self.ax = self.fig.add_subplot(111)
        self.camera = Camera(self.fig)
        if not hasattr(self.model, 'distance'):
            # Hook the distance layer to the model under model.distance.
            self.model.distance = self.model.layers[self.distance_layer_index]

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_train_end(self, logs={}):
        if self.make_gif:
            gif_from_dir(directory=self.save_dir,
                         prefix=self.prefix,
                         duration=1.0 / self.fps)
        if self.snap and self.make_mp4:
            animation = self.camera.animate()
            vid = os.path.join(self.save_dir, f'{self.prefix}animation.mp4')
            if self.verbose:
                print(f'Saving mp4 under {vid}.')
            animation.save(vid, fps=self.fps, dpi=self.dpi)


class VisPointProtos(VisWeights):
    """Visualization of prototypes.

    .. TODO::
        Still in Progress.
    """
    def __init__(self, prototype_layer, **kwargs):
        super().__init__(**kwargs)
        self.prototype_layer = prototype_layer
        self.title = 'Point Prototypes Visualization'
        self.data_scatter_settings = {
            'marker': 'o',
            's': 30,
            'edgecolor': 'k',
            'cmap': self.cmap,
        }
        self.protos_scatter_settings = {
            'marker': 'D',
            's': 50,
            'edgecolor': 'k',
            'cmap': self.cmap,
        }

    def on_epoch_begin(self, epoch, logs={}):
        if self._skip_epoch(epoch):
            return True

        self._clean_and_setup_ax()

        protos = self.prototype_layer.prototypes.numpy()
        labels = self.model.distance.prototype_labels.numpy()

        if self.project_protos:
            protos = self.model.projection(protos).numpy()

        color_map = color_scheme(n=len(set(labels)),
                                 cmap=self.cmap,
                                 zero_indexed=True)
        # TODO Get rid of the assumption y values in [0, num_of_classes]
        label_colors = [color_map[l] for l in labels]

        if self.data is not None:
            x, y = self.data
            # TODO Get rid of the assumption y values in [0, num_of_classes]
            y_colors = [color_map[l] for l in y]
            # x = self.model.projection(x)
            if not isinstance(x, np.ndarray):
                x = x.numpy()

            # Plot data points.
            self.ax.scatter(x[:, 0],
                            x[:, 1],
                            c=y_colors,
                            **self.data_scatter_settings)

            # Paint decision regions.
            if self.voronoi:
                border = self.border
                resolution = self.resolution
                x = np.vstack((x, protos))
                x_min, x_max = x[:, 0].min(), x[:, 0].max()
                y_min, y_max = x[:, 1].min(), x[:, 1].max()
                x_min, x_max = x_min - border, x_max + border
                y_min, y_max = y_min - border, y_max + border
                try:
                    xx, yy = np.meshgrid(
                        np.arange(x_min, x_max, (x_max - x_min) / resolution),
                        np.arange(y_min, y_max, (x_max - x_min) / resolution))
                except ValueError as ve:
                    print(ve)
                    raise ValueError(f'x_min: {x_min}, x_max: {x_max}. '
                                     f'x_min - x_max is {x_max - x_min}.')
                except MemoryError as me:
                    print(me)
                    raise ValueError('Too many points. '
                                     'Try reducing the resolution.')
                mesh_input = np.c_[xx.ravel(), yy.ravel()]

                # Predict mesh labels.
                if self.project_mesh:
                    mesh_input = self.model.projection(mesh_input)

                # d = self.model(mesh_input)  # WRONG! doesn't stop gradients
                d = self.model.predict(mesh_input, batch_size=len(mesh_input))
                # d = self.model.predict(mesh_input,
                #                        batch_size=mesh_input.shape[0])
                if self.ignore_last_output_row:
                    d = d[:-1]
                y_pred = self.model.competition(d).numpy()
                y_pred = y_pred.reshape(xx.shape)

                # Plot voronoi regions.
                self.ax.contourf(xx, yy, y_pred, cmap=self.cmap, alpha=0.35)

                self.ax.set_xlim(left=x_min + 0, right=x_max - 0)
                self.ax.set_ylim(bottom=y_min + 0, top=y_max - 0)

        # Plot prototypes.
        self.ax.scatter(protos[:, 0],
                        protos[:, 1],
                        c=label_colors,
                        **self.protos_scatter_settings)

        # self._show_and_save(epoch)

    def on_epoch_end(self, epoch, logs={}):
        self._display_logs(self.ax, epoch, logs)
        self._show_and_save(epoch)
