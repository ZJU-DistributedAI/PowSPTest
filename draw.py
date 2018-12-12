import matplotlib.pyplot as plt
import numpy as np
import mdpsol
import numpy as np

def draw_figure2():
    x,y001,y01 = mdpsol.get_figure2()

    plt.title('Result Analysis')
    plt.plot(x, y001, color='green', label='rs 1%')
    plt.plot(x, y01, color='red', label='rs 10%')
    plt.plot(x, x, color='blue', label='honest')
    plt.legend()
    plt.xlabel('alpha')
    plt.ylabel('rrel')
    plt.show()

def draw_figure3():
    rs,rho01,rho03 = mdpsol.get_figure3()

    plt.title('Result Analysis')
    plt.plot(rs, rho01, color='green', label='alpha 0.1')
    plt.plot(rs, rho03, color='red', label='alpha 0.3')
    plt.legend()
    plt.xlabel('rs')
    plt.ylabel('rrel')
    plt.show()

def draw_figure4():
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator

    x,y,z = mdpsol.get_figure4()
    z = z[:-1, :-1]
    levels = MaxNLocator(nbins=25).tick_values(z.min(), z.max())

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('plasma')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, ax  = plt.subplots(
        nrows=1)

    im = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax)
    ax.set_title('result')

    fig.tight_layout()

    plt.show()

def draw_figure6():
    alpha , vd0,vda,vdd = mdpsol.get_figure6()
    plt.title('Result Analysis')
    plt.semilogy(alpha, vd0,  color='green', label='cm = 0')
    plt.semilogy(alpha, vda, color='red', label='cm = alpha')
    plt.semilogy(alpha, vdd, color='blue', label='Δvd')
    plt.legend()
    plt.xlabel('alpha')
    plt.ylabel('vd')
    plt.show()

def draw_figure7():
    rs,vd1,vd3= mdpsol.get_figure7()
    plt.title('Result Analysis')
    plt.semilogy(rs, vd1,  color='green', label='alpha = 0.1')
    plt.semilogy(rs, vd3, color='red', label='alpha = 0.3')
    plt.legend()
    plt.xlabel('rs')
    plt.ylabel('vd')
    plt.show()

def get_table3():
    mdpsol.get_table3()

def draw_figure8():
    gamma,alpha,k,p0,p5,p1 = mdpsol.get_figure8()
    colors = ['blue','green','red','olive','pink','cyan']

    def example_plot(ax,gamma,ks,ps):
        for i in range(len(ks)):
            ax.semilogy(alpha,ps[i], color=colors[i], label='k = '+str(ks[i]))
        ax.set_xlabel('alpha')
        ax.set_ylabel('vd')
        ax.set_title('gamma='+str(gamma))
        ax.legend()

    fig, axs = plt.subplots(ncols=3, constrained_layout=False)
    example_plot(axs[0],0,k,p0)
    example_plot(axs[1],0.5,k,p5)
    example_plot(axs[2],1,k,p1)
    plt.show()

def draw_figure9():
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    x,y,z = mdpsol.get_figure9()
    z = z[:-1, :-1]
    levels = MaxNLocator(nbins=25).tick_values(z.min(), z.max())

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('plasma')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, ax  = plt.subplots(
        nrows=1)

    im = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax)
    ax.set_title('result')
    fig.tight_layout()
    plt.show()

draw_figure9()