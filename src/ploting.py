import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def plot_board(b, filename=None):
    fig = plt.figure()
    ax = fig.gca()
    dim = b.shape[1]

    ticks = np.arange(-2,dim*2,2)+1
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    empty_string_labels = ['']*len(ticks)
    ax.set_xticklabels(empty_string_labels)
    ax.set_yticklabels(empty_string_labels)

    #subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
    #            hspace = 0, wspace = 0)
    #margins(0,0)
    #gca().xaxis.set_major_locator(NullLocator())
    #gca().yaxis.set_major_locator(NullLocator())
    #ax.use_sticky_edges = True
    #ax.margins(0.04,0.04)
    print(plt.xlim(-1,25))
    print(plt.ylim(1,27))

    x, y = np.where(b[1] == 1)
    s = [ 100 for _ in range(len(x)) ]
    y *= 2
    x *= 2
    x = dim*2 - x
    plt.scatter(y, x, s=s, marker='x', color='r')
    x, y = np.where(b[0] == 1)
    s = [ 100 for _ in range(len(x)) ]
    y *= 2
    x *= 2
    x = dim*2 - x
    plt.scatter(y, x, s=s, marker='o', color='b')
    x, y = np.where(b[0] == 2)
    s = [ 100 for _ in range(len(x)) ]
    y *= 2
    x *= 2
    x = dim*2 - x
    plt.scatter(y, x, s=s, marker='o', color='#2deb00ff')
    ax.set_aspect('equal')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    plt.grid()
    plt.show()
    #plt.tick_params(axis='xy', bottom=False, top=False, labelbottom=False)
    print("filename: ", filename)

    if filename is not None:
        print('saving...')
        plt.savefig(filename, format='svg', bbox_inches='tight')

def plot_board2(b, filename=None):
    fig, ax = plt.subplots()
    dim = b.shape[1]


    dim = b.shape[1]
    im = ax.imshow(b[0]-b[1], cmap='bwr')

    ax.set_yticks(np.arange(dim))
    ax.set_xticks(np.arange(dim))

    #x, y = np.where(b[1] == 1)
    #s = [ 100 for _ in range(len(x)) ]
    #plt.scatter(y, x, s=s, marker='o', color='b')
    #x, y = np.where(b[0] == 1)
    #plt.scatter(y, x, s=s, marker='x', color='r')
    ax.set_aspect('equal')
    if filename is not None:
        plt.savefig(filename, format='svg', bbox_inches='tight')

def plot_pi(pi, b, annotate=False, filename=None):
    fig, ax = plt.subplots()
    dim = b.shape[2]

    pi = pi.reshape((dim, dim))
    im = ax.imshow(pi, cmap='Blues')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(im, cax=cax)
    #cbar.ax.set_ylabel("probability",rotation=-90, va='bottom')
    ax.set_yticks(np.arange(dim))
    ax.set_xticks(np.arange(dim))

    threshold = im.norm(pi.max())/2.

    if annotate and filename is not None:
        print("anotating")
        #ax.annotate(, xy=(0,1), xycoords='figure fraction')
        ax.annotate(annotate, xy=(0.12,0.92), xycoords='figure fraction')

    print(pi.max())
    print(threshold)

    for i in range(dim):
        for j in range(dim):
            if b[0][i,j] == 1:
                text = ax.text(j, i, "x", ha="center", va="center", color="w")
                continue
            if b[1][i,j] == 1:
                text = ax.text(j, i, "o", ha="center", va="center", color="w")
                continue

            #if im.norm(pi[i,j]) > threshold:
            #    c = 'w'
            #else:
            #    c = 'xkcd:black'

            #if pi[i,j] > 0.01:
            #    s = "{0:.2f}".format(pi[i, j])
            #else:
            #    s = ""
            #if s:
            #    text = ax.text(j, i, s, fontsize=7, ha="center", va="center", color=c)

    #kw = dict(horizontalalignment="center",
    #          verticalalignment="center")
    #kw.update(textkw)

    if filename is not None:
        plt.savefig(filename, format='svg', bbox_inches='tight')
        print("Saving to: {}".format(filename))
