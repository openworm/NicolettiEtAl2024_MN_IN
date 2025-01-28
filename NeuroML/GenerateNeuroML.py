from neuroml import NeuroMLDocument
from neuroml.utils import component_factory

from pyneuroml import pynml
from pyneuroml.xppaut import parse_script
from pprint import pprint

from neuroml import GateHHRates
from neuroml import IncludeType
import sympy
from sympy.parsing.sympy_parser import parse_expr
import math


def generate_nmllite(
    cell,
    duration=11000,
    config="IClamp",
    parameters=None,
    channels_to_include=[],
    color=None,
):
    from neuromllite import Cell, InputSource

    # from neuromllite.NetworkGenerator import *
    from neuromllite.utils import create_new_model

    reference = "%s_%s" % (config, cell)

    cell_id = "%s" % cell
    cell_nmll = Cell(id=cell_id, neuroml2_source_file="%s.cell.nml" % (cell))

    ################################################################################
    ###   Add some inputs

    if "IClamp" in config:
        if not parameters:
            parameters = {}

        """
        input_source = InputSource(
            id="iclamp_0",
            neuroml2_input="PulseGenerator",
            parameters={
                "amplitude": "stim_amp",
                "delay": "stim_delay",
                "duration": "stim_duration",
            },
        )"""

    else:
        if not parameters:
            parameters = {}
            parameters["average_rate"] = "100 Hz"
            parameters["number_per_cell"] = "10"

        input_source = InputSource(
            id="pfs0",
            neuroml2_input="PoissonFiringSynapse",
            parameters={
                "average_rate": "average_rate",
                "synapse": syn_exc.id,
                "spike_target": "./%s" % syn_exc.id,
            },
        )

    sim, net = create_new_model(
        reference,
        duration,
        dt=0.025,  # ms
        temperature=34,  # degC
        default_region="Worm",
        parameters=parameters,
        cell_for_default_population=cell_nmll,
        color_for_default_population=color,
        input_for_default_population=None,
    )

    net.parameters = {}

    amps = [15 + 2 * i for i in range(11)]
    net.populations[0].size = len(amps)
    net.input_sources = []
    net.inputs = []

    from neuromllite import InputSource, Input

    for i in amps:
        ins = InputSource(
            id="iclamp_stim_%s" % str(i).replace("-", "min"),
            neuroml2_input="PulseGenerator",
            parameters={
                "amplitude": "%spA" % i,
                "delay": "1000ms",
                "duration": "5000ms",
            },
        )
        net.input_sources.append(ins)
        net.inputs.append(
            Input(
                id="input_%s" % ins.id,
                input_source=ins.id,
                population=net.populations[0].id,
                cell_ids=[amps.index(i)],
            )
        )

    net.to_json_file()

    sim.record_variables = {"caConc": {net.populations[0].id: "*"}}
    """
    for c in channels_to_include:
        not_on_rmd = ["kvs1", "kqt3", "egl2"]
        if c == "ca":
            c = "sk"

        if c != "egl36" and cell != "AWCon" and not (c in not_on_rmd and cell == "RMD"):
            sim.record_variables["biophys/membraneProperties/%s_chans/gDensity" % c] = {
                "all": "*"
            }
            sim.record_variables["biophys/membraneProperties/%s_chans/iDensity" % c] = {
                "all": "*"
            }
        if (
            c != "leak"
            and c != "nca"
            and not (c == "egl36" and cell == "AWCon")
            and not (c in not_on_rmd and cell == "RMD")
        ):
            sim.record_variables[
                "biophys/membraneProperties/%s_chans/%s/m/q" % (c, c)
            ] = {"all": "*"}
        if (
            c != "leak"
            and c not in ["nca", "kir", "sk", "egl36", "kqt3", "egl2", "irk"]
            and not (c in not_on_rmd and cell == "RMD")
        ):
            sim.record_variables[
                "biophys/membraneProperties/%s_chans/%s/h/q" % (c, c)
            ] = {"all": "*"}

        if cell == "AWCon" and c in ["kqt3"]:
            sim.record_variables[
                "biophys/membraneProperties/%s_chans/%s/s/q" % (c, c)
            ] = {"all": "*"}
            sim.record_variables[
                "biophys/membraneProperties/%s_chans/%s/w/q" % (c, c)
            ] = {"all": "*"}"""

    sim.to_json_file()

    return sim, net


def create_cell(
    cell_id, duration, channels_to_include, conductances, cell_params, color
):
    # Create the nml file and add the ion channels
    cell_doc = NeuroMLDocument(id=cell_id, notes="A cell from Nicoletti et al. 2024")
    cell_fn = "%s.cell.nml" % cell_id

    # Define a cell
    cell = cell_doc.add(
        "Cell", id=cell_id, notes="%s cell from Nicoletti et al. 2019" % cell_id
    )
    """
        volume_um3 = xpps[cell_id]["parameters"]["vol"]
        diam = 1.7841242
        end_area = math.pi * diam * diam / 4
        length = volume_um3 / end_area
        surface_area_curved = length * math.pi * diam"""

    surf = cell_params["surf"]
    # vol = 7.42e-12  # total volume
    L = math.sqrt(surf / math.pi)
    rsoma = L * 1e4

    cell.add_segment(
        prox=[0, 0, 0, rsoma],
        dist=[0, rsoma, 0, rsoma],
        name="soma",
        parent=None,
        fraction_along=1.0,
        seg_type="soma",
    )

    cell.add_membrane_property("SpikeThresh", value="0mV")

    cell.set_specific_capacitance("%s uF_per_cm2" % (cell_params["cm"]))

    cell.set_init_memb_potential("-40mV")

    # This value is not really used as it's a single comp cell model
    cell.set_resistivity("0.1 kohm_cm")

    for channel_id in sorted(channels_to_include):
        density_scaled = (cell_params[channel_id] * 1e-9) / (surf)

        print(cell_params)
        erev = cell_params["eleak"]
        ion = "non_specific"

        if channel_id in ["egl19"]:
            erev = 60
            ion = "ca"
        if channel_id in ["irk"]:
            erev = -80
            ion = "k"
        if channel_id in ["nca"]:
            erev = 30
        cell.add_channel_density(
            cell_doc,
            cd_id="%s_chans" % channel_id,
            cond_density="%s S_per_cm2" % density_scaled,
            erev="%smV" % erev,
            ion=ion,
            ion_channel="%s" % channel_id,
            ion_chan_def_file="%s.channel.nml" % channel_id,
        )

    """
        cell_doc.includes.append(IncludeType(href="CaDynamics.nml"))
        # <species id="ca" ion="ca" concentrationModel="CaDynamics" initialConcentration="1e-4 mM" initialExtConcentration="2 mM"/>
        species = component_factory(
            "Species",
            id="ca",
            ion="ca",
            concentration_model="CaDynamics_%s" % cell_id,
            initial_concentration="5e-5 mM",
            initial_ext_concentration="2 mM",
        )

        cell.biophysical_properties.intracellular_properties.add(species)"""

    cell.info(show_contents=True)

    cell_doc.validate(recursive=True)
    pynml.write_neuroml2_file(nml2_doc=cell_doc, nml2_file_name=cell_fn, validate=True)

    sim, net = generate_nmllite(
        cell_id,
        duration=duration,
        config="IClamp",
        parameters=None,
        channels_to_include=channels_to_include,
        color=color,
    )

    ################################################################################
    ###   Run in some simulators

    from neuromllite.NetworkGenerator import check_to_generate_or_run
    import sys

    check_to_generate_or_run(sys.argv, sim)


if __name__ == "__main__":
    all = {}

    all["AVAL"] = {"color": "0.5 1 1"}
    all["AVAL"]["cell_params"] = {
        "surf": 1123.84e-8
    }  # surface in cm^2 form neuromorpho AIYL
    all["AVAL"]["conductances"] = ["egl19", "leak", "irk", "nca", "eleak", "cm"]
    all["AVAL"]["g0"] = [0.104385, 0.150164, 0.1, 0, -39, 0.859551]

    for cell in all:
        cell_params = all[cell]["cell_params"]
        conductances = all[cell]["conductances"]
        g0 = all[cell]["g0"]

        for a in zip(conductances, g0):
            print(f"Setting {a[0]} = {a[1]} for {cell}")
            cell_params[a[0]] = a[1]

        chans = []
        for c in conductances:
            if "eleak" not in c and "cm" not in c:
                chans.append(c)

        create_cell(
            cell_id=cell,
            duration=11000,
            channels_to_include=chans,
            conductances=conductances,
            cell_params=cell_params,
            color=all[cell]["color"],
        )
