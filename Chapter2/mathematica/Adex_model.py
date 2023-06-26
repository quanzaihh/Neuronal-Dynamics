import brian2 as b2
from neurodynex3.adex_model import AdEx
from neurodynex3.tools import plot_tools, input_factory
from matplotlib import pyplot as plt

current = input_factory.get_step_current(10, 250, 1. * b2.ms, 65.0 * b2.pA)
state_monitor, spike_monitor = AdEx.simulate_AdEx_neuron(I_stim=current, simulation_time=400 * b2.ms, tau_m=5*b2.ms,
                                                         tau_w=100*b2.ms, a=-0.5*b2.nS, b=7.*b2.pA, v_reset=-46*b2.mV)
plot_tools.plot_voltage_and_current_traces(state_monitor, current)
plt.show()
print("nr of spikes: {}".format(spike_monitor.count[0]))
# AdEx.plot_adex_state(state_monitor)