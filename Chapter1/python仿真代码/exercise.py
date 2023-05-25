import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
from neurodynex3.leaky_integrate_and_fire import LIF
from neurodynex3.tools import *
from neurodynex3.leaky_integrate_and_fire.LIF import *

# t_spikes = [(i+1)*10 for i in range(30)]
# step_current = input_factory.get_spikes_current_my(
#     t_spikes=t_spikes, unit_time=b2.ms,
#     amplitude=12 * b2.namp)
step_current = input_factory.get_spikes_ramdon_current(frequent=60, windows=300, Refractory=2, unit_time=b2.ms, amplitude=12*b2.namp)
# run the LIF model
(state_monitor, spike_monitor) = simulate_LIF_neuron(input_current=step_current, simulation_time=300 * b2.ms)

# plot the membrane voltage
plot_tools.plot_voltage_and_current_traces(state_monitor, step_current,
                                           title="Step current", firing_threshold=FIRING_THRESHOLD)
print("nr of spikes: {}".format(len(spike_monitor.t)))
plt.show()