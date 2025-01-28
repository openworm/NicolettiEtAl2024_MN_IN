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


colors = {"RIM": "0.5 1 1"}
cell_params = {}

cell = "RIM"
cell_params[cell] = {"surf": 103.34e-8}  # surface in cm^2 form neuromorpho AIYL

conductances = [
    "shl1",
    "egl2",
    "irk",
    "cca1",
    "unc2",
    "egl19", 
    "leak",
    "eleak", 
    "cm"
]

# conductances in S/cm^2: SHL1, EGL2, IRK, CCA1, unc2, egl19, LEAK,eleak, cm

g=[0.0009048750067326097,0.0001411644285181245,0.0003272854640954744,0.0008451919806776876,9.676795045480941e-05,0.00032005818627638106,9.676795045480941e-05,-50,1.5]




for a in zip(conductances, g0):
    print(f"Setting {a[0]} = {a[1]} for {cell}")
    cell_params[cell][a[0]] = a[1]


def generate_nmllite(
    cell,
    duration=11000,
    config="IClamp",
    parameters=None,
    stim_delay=1000,
    stim_duration=5000,
    channels_to_include=[],
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
            parameters["stim_amp"] = "30pA"
            parameters["stim_delay"] = "%sms" % stim_delay
            parameters["stim_duration"] = "%sms" % stim_duration

        input_source = InputSource(
            id="iclamp_0",
            neuroml2_input="PulseGenerator",
            parameters={
                "amplitude": "stim_amp",
                "delay": "stim_delay",
                "duration": "stim_duration",
            },
        )

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
        color_for_default_population=colors[cell],
        input_for_default_population=input_source,
    )
    sim.record_variables = {"caConc": {"all": "*"}}
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
            and c not in ["nca", "kir", "sk", "egl36", "kqt3", "egl2"]
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
            ] = {"all": "*"}

    sim.to_json_file()

    return sim, net


def create_cells(channels_to_include, duration=700, stim_delay=310, stim_duration=500):
    for cell_id in cell_params.keys():
        # Create the nml file and add the ion channels
        cell_doc = NeuroMLDocument(
            id=cell_id, notes="A cell from Nicoletti et al. 2024"
        )
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

        surf = cell_params[cell_id]["surf"]
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

        cell.set_specific_capacitance("%s uF_per_cm2" % (cell_params[cell_id]["cm"]))

        cell.set_init_memb_potential("-89.57mV")

        # This value is not really used as it's a single comp cell model
        cell.set_resistivity("0.1 kohm_cm")

        for channel_id in channels_to_include:
            density_scaled = (cell_params[cell_id][channel_id] * 1e-9) / (surf)

            print(cell_params[cell_id])
            cell.add_channel_density(
                cell_doc,
                cd_id="%s_chans" % channel_id,
                cond_density="%s S_per_cm2" % density_scaled,
                erev="%smV" % cell_params[cell_id]["eleak"],
                ion="non_specific",
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
        pynml.write_neuroml2_file(
            nml2_doc=cell_doc, nml2_file_name=cell_fn, validate=True
        )

        sim, net = generate_nmllite(
            cell_id,
            duration=duration,
            config="IClamp",
            parameters=None,
            stim_delay=stim_delay,
            stim_duration=stim_duration,
            channels_to_include=channels_to_include,
        )

        ################################################################################
        ###   Run in some simulators

        from neuromllite.NetworkGenerator import check_to_generate_or_run
        import sys

        check_to_generate_or_run(sys.argv, sim)


if __name__ == "__main__":
    create_cells(
        channels_to_include=["leak", "unc103", "irk", "egl19", "nca"],
   
        duration=11000,
        stim_delay=500,
        stim_duration=2000,
    )


103.34e-8