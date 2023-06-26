import brian2 as b2
import matplotlib.pyplot as plt
import neurodynex3.exponential_integrate_fire.exp_IF as exp_IF
from neurodynex3.tools import plot_tools, input_factory
from tqdm import tqdm
import math

i=0  #change i and find the value that goes into min_amp
durations = [1,   2,    5,  10,   20,   50, 100]
min_amp =   [8.58, 4.42, 1.93, 1.10, 0.70, 0.48, 0.43]

# t=durations[i]
# I_amp = min_amp[i]*b2.namp
# title_txt = "I_amp={}, t={}".format(I_amp, t*b2.ms)
#
# input_current = input_factory.get_step_current(t_start=10, t_end=10+t-1, unit_time=b2.ms, amplitude=I_amp)
#
# state_monitor, spike_monitor = exp_IF.simulate_exponential_IF_neuron(I_stim=input_current, simulation_time=(t+20)*b2.ms)

# plot_tools.plot_voltage_and_current_traces(state_monitor, input_current,
#                                            title=title_txt, firing_threshold=exp_IF.FIRING_THRESHOLD_v_spike,
#                                           legend_location=2)
# print("nr of spikes: {}".format(spike_monitor.count[0]))

min_amp = [math.log(i) for i in min_amp]
durations = [math.log(i) for i in durations]
plt.plot(durations, min_amp)
plt.title("Strength-Duration curve")
plt.xlabel("t [ms]")
plt.ylabel("min amplitude [nAmp]")
plt.show()



