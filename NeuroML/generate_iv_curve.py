import sys
import pyneuroml.analysis
from pyneuroml.analysis import generate_current_vs_frequency_curve

from matplotlib import pyplot as plt
from pyelectro.analysis import max_min

nogui = "-nogui" in sys.argv  # Used to supress GUI in tests for Travis-CI
print(dir(pyneuroml.analysis))

for cell in ["AVAL", "AVAR"]:
    generate_current_vs_frequency_curve(
        "%s.cell.nml" % cell,
        cell,
        start_amp_nA=-0.03,
        end_amp_nA=0.04,
        step_nA=0.01,
        analysis_duration=1000,
        analysis_delay=100,
        pre_zero_pulse=100,
        post_zero_pulse=500,
        plot_voltage_traces=not nogui,
        plot_if=not nogui,
        plot_iv=not nogui,
        save_if_data_to="iv_%s.dat" % cell,
        show_plot_already=False,
    )

if not nogui:
    plt.show()
