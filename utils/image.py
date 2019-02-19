from matplotlib.pyplot import imsave


def scale_in(image):
    return (image+1.0)/2.0

def scale_out(image):
    return (image*2.0)-1.0

def save(savedir, image, row, column):
    plot_column_list = []
    _plot_row = np.split(image, row)
    for _plot in _plot_row:
        _plot_column = np.split(np.squeeze(_plot), column)
        plot_column = np.squeeze(np.concatenate(_plot_column, axis=1))
        plot_column_list.append(plot_column)
    plot = np.concatenate(plot_column_list, axis=1)

    imsave(path.join(savedir, 'result.jpg'), plot)
