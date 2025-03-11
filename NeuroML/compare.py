from pyneuroml import pynml
from pyneuroml.plot.Plot import generate_plot
from matplotlib import pyplot as plt

plots = {
    "Membrane potentials": ["Soma.v.dat", "../Soma.si.dat"],
    "Ca conc": ["Soma.ca.dat", "../Soma.ca.dat"],
    "Channel activation": ["Soma.chans.dat", "../Soma.chans.dat"],
    "Ca rev pot": ["Soma.eca.dat", "../Soma.eca.dat"],
}

for p in plots:
    files = plots[p]

    times = []
    xs = []
    labels = []
    linewidths = []

    d, indices = pynml.reload_standard_dat_file(files[0])
    for i in indices:
        if i != "t":
            times.append(d["t"])
            xs.append(d[i])
            labels.append("jnml %i" % i if len(d) > 2 else "jnml")
            linewidths.append(3)

    d, indices = pynml.reload_standard_dat_file(files[1])

    for i in indices:
        if i != "t":
            times.append(d["t"])
            xs.append(d[i])
            labels.append("nrn %i" % i if len(d) > 2 else "nrn")
            linewidths.append(1)

    generate_plot(
        times,
        xs,
        p,
        labels=labels,
        linewidths=linewidths,
        show_plot_already=False,
    )


plt.show()
