import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
from neurodynex3.leaky_integrate_and_fire import LIF
from neurodynex3.tools import *
from neurodynex3.leaky_integrate_and_fire.LIF import *
from tqdm import tqdm

spike = []
frequency = []
for i in tqdm(range(90)):
    sinusoidal_current = input_factory.get_sinusoidal_current(
        500, 1500, unit_time=0.1 * b2.ms,
        amplitude=2.5 * b2.namp, frequency=(50 + i * 5) * b2.Hz, direct_current=2. * b2.namp)

    # step_current = input_factory.get_spikes_ramdon_current(frequent=60, windows=300, Refractory=2, unit_time=b2.ms,
    #                                                        amplitude=12 * b2.namp)
    # run the LIF model
    (state_monitor, spike_monitor) = simulate_LIF_neuron(input_current=sinusoidal_current, simulation_time=200 * b2.ms)

    # plot the membrane voltage
    # plot_tools.plot_voltage_and_current_traces(state_monitor, sinusoidal_current,
    #                                            title="Step current", firing_threshold=FIRING_THRESHOLD)
    # print("nr of spikes: {}".format(len(spike_monitor.t)))
    # plt.show()

    frequency.append(50 + i * 5)
    spike.append(len(spike_monitor.t))

plt.plot(frequency, spike)
plt.xlabel("frequent")
plt.ylabel("spike_num")
plt.savefig("../增益和频率的关系.pdf")
