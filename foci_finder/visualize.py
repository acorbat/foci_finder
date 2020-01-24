import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from . import foci_analysis as fa


def visualizer(stack, tp_dict, cell_mask, LoG_size=None, min_area=0):
    fig = plt.figure(constrained_layout=True, figsize=(15, 30))
    spec = fig.add_gridspec(ncols=2, nrows=2, height_ratios=[5, 1])
    axs = []
    axs.append(fig.add_subplot(spec[0, 0]))
    axs.append(fig.add_subplot(spec[0, 1]))
    axs.append(fig.add_subplot(spec[1, :]))

    ini_z = stack.shape[1]//2
    ini_t = 0
    ini_thresh = tp_dict[ini_t]

    subplot = SubPlot(axs[0], stack, cell_mask, z=ini_z, t=ini_t,
                      thresh=ini_thresh, LoG_size=LoG_size, min_area=min_area)
    simple_subplot = SimpleSubPlot(axs[1], stack, z=ini_z, t=ini_t)

    callback = Index(subplot, simple_subplot,
                     stack.shape[1]-1, stack.shape[0]-1)
    callback.cur_z = ini_z
    callback.cur_t = ini_t

    plt.sca(axs[2])

    axs[2].axis('off')

    axprev_z = plt.axes([0.6, 0.05, 0.15, 0.075])
    axnext_z = plt.axes([0.75, 0.05, 0.15, 0.075])
    bnext_z = Button(axnext_z, 'Next Z')
    bnext_z.on_clicked(callback.next_z)
    bprev_z = Button(axprev_z, 'Previous Z')
    bprev_z.on_clicked(callback.prev_z)

    axprev_t = plt.axes([0.3, 0.05, 0.15, 0.075])
    axnext_t = plt.axes([0.45, 0.05, 0.15, 0.075])
    bnext_t = Button(axnext_t, 'Next T')
    bnext_t.on_clicked(callback.next_t)
    bprev_t = Button(axprev_t, 'Previous T')
    bprev_t.on_clicked(callback.prev_t)

    axthresh = plt.axes([0.5, 0.15, 0.3, 0.03],
                        facecolor='lightgoldenrodyellow')
    sthresh = Slider(axthresh, 'Thresh',
                     valmin=-0.1,
                     valmax=0.5,
                     valinit=ini_thresh,
                     valstep=0.005)

    def change_threshold(val):
        subplot.thresh = val
        subplot.relabel()
        subplot.update()

    sthresh.on_changed(change_threshold)

    def select_threshold(event):
        print('Setting threshold to: %s' % subplot.thresh)
        nonlocal tp_dict
        tp_dict[subplot.t] = subplot.thresh.item()

    axchoose = plt.axes([0.1, 0.05, 0.1, 0.075])
    bchoose = Button(axchoose, 'Select')
    bchoose.on_clicked(select_threshold)

    plt.show()

    return tp_dict


class SubPlot(object):

    def __init__(self, axs, stack, cell_mask, z=0, t=0, thresh=0,
                 LoG_size=None, min_area=0):
        self.axs = axs
        self.stack = stack
        self.cell_mask = cell_mask
        self.z = z
        self.t = t
        self.thresh = thresh
        self.LoG_size = LoG_size
        self.min_area = min_area
        self.min_intensity = self.stack.min()
        self.max_intensity = self.stack.max()
        self.max_intensity = self.max_intensity / 6 if self.max_intensity / 6 > self.min_intensity else self.max_intensity
        self.labeled = fa.manual_find_foci(self.stack[t], self.thresh,
                                           LoG_size=self.LoG_size,
                                           min_area=self.min_area)
        plt.sca(self.axs)

        self.img = self.axs.imshow(self.stack[self.t][self.z], cmap='Greys_r')
        self.axs.axis('off')

        self.cont_cell = self.axs.contour(self.cell_mask[self.t][self.z],
                                          colors='r', linewidths=0.4)

        self.cont = self.axs.contour(self.labeled[self.z] > 0, colors='yellow',
                                     linewidths=0.2, alpha=0.4)

        self.set_title()
        plt.draw()

    def update(self):
        plt.sca(self.axs)
        self.img.set_data(self.stack[self.t][self.z])
        self.set_title()
        for cont in self.cont_cell.collections:
            cont.remove()
        self.cont_cell = self.axs.contour(self.cell_mask[self.t][self.z],
                                          colors='r', linewidths=0.4)
        for cont in self.cont.collections:
            cont.remove()
        self.cont = self.axs.contour(self.labeled[self.z] > 0, colors='yellow',
                                     linewidths=0.2, alpha=0.4)
        plt.draw()

    def relabel(self):
        print('relabeling for t: %s; thresh: %s' % (self.t, self.thresh))
        self.labeled = fa.manual_find_foci(self.stack[self.t], self.thresh,
                                           LoG_size=self.LoG_size,
                                           min_area=self.min_area)
        self.update()

    def set_title(self):
        self.axs.set_title('z = %s; t = %s' % (self.z, self.t))


class SimpleSubPlot(object):

    def __init__(self, axs, stack, z=0, t=0):
        self.axs = axs
        self.stack = stack
        self.z = z
        self.t = t
        self.min_intensity = self.stack.min()
        self.max_intensity = self.stack.max()
        self.max_intensity = self.max_intensity / 6 if self.max_intensity / 6 > self.min_intensity else self.max_intensity
        plt.sca(self.axs)

        self.img = self.axs.imshow(self.stack[self.t][self.z], cmap='Greys_r')
        self.axs.axis('off')
        self.set_title()
        plt.draw()

    def update(self):
        plt.sca(self.axs)
        self.img.set_data(self.stack[self.t][self.z])
        self.set_title()
        plt.draw()

    def set_title(self):
        self.axs.set_title('z = %s; t = %s' % (self.z, self.t))


class Index(object):

    def __init__(self, subplot, simple_subplot, z_len, t_len):
        self.subplot = subplot
        self.simple_subplot = simple_subplot

        self.cur_z = 0
        self.max_z = z_len

        self.cur_t = 0
        self.max_t = t_len

    def next_z(self, event):
        print('Next Z')
        self.cur_z= min(self.cur_z + 1, self.max_z)

        self.subplot.z = self.cur_z
        self.simple_subplot.z = self.cur_z
        self.subplot.update()
        self.simple_subplot.update()

    def prev_z(self, event):
        print('Prev Z')
        self.cur_z = max(self.cur_z - 1, 0)

        self.subplot.z = self.cur_z
        self.simple_subplot.z = self.cur_z
        self.subplot.update()
        self.simple_subplot.update()

    def next_t(self, event):
        print('Next T')
        self.cur_t = min(self.cur_t + 1, self.max_t)

        self.subplot.t = self.cur_t
        self.simple_subplot.t = self.cur_t
        self.subplot.relabel()
        self.subplot.update()
        self.simple_subplot.update()

    def prev_t(self, event):
        print('Prev T')
        self.cur_t = max(self.cur_t - 1, 0)

        self.subplot.t = self.cur_t
        self.simple_subplot.t = self.cur_t
        self.subplot.relabel()
        self.subplot.update()
        self.simple_subplot.update()
