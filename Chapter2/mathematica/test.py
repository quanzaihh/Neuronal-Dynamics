from matplotlib import pyplot as plt
import data
import neurodynex3.exponential_integrate_fire.exp_IF as exp_IF
from neurodynex3.tools import plot_tools, input_factory
import brian2 as b2
import numpy as np

bits, signal = data.Bspk_code(size=2, plot=True)
tmp = np.zeros((len(signal)+1, 1)) * b2.amp
tmp[:-1, 0] = signal * b2.namp * 0.5
curr = b2.TimedArray(tmp, dt=1. * b2.ms)

state_monitor, spike_monitor = exp_IF.simulate_exponential_IF_neuron(I_stim=curr, simulation_time=(len(signal)+100)*b2.ms)

plot_tools.plot_voltage_and_current_traces(state_monitor, curr,
                                           title="nothing", firing_threshold=exp_IF.FIRING_THRESHOLD_v_spike,
                                          legend_location=2)
plt.show()
