from pyneuroml import pynml
from pyneuroml.plot.Plot import generate_plot
from matplotlib import pyplot as plt

plots = {'Membrane potentials':['Soma.v.dat', '../Soma.si.dat'],
         'Ca conc':['Soma.ca.dat', '../Soma.ca.dat'],
         }

for p in plots:
    files = plots[p]

    d, i = pynml.reload_standard_dat_file(files[0])

    times_jnml = d['t']
    xs_jnml = d[0]

    d, i = pynml.reload_standard_dat_file(files[1])

    times_nrn = d['t']
    xs_nrn = d[0]

    generate_plot(
            [times_nrn,times_jnml],
            [xs_nrn, xs_jnml],
            p,
            labels=["nrn", "jnml"],
            linewidths=[2, 1], 
            show_plot_already=False,)
        

plt.show()