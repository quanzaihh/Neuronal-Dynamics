import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
from neurodynex3.leaky_integrate_and_fire import LIF
from neurodynex3.tools import *
from neurodynex3.leaky_integrate_and_fire.LIF import *
from tqdm import tqdm


def Simulate(frequent):
    step_current = input_factory.get_spikes_ramdon_current(frequent=60, windows=500, Refractory=2, unit_time=b2.ms,
                                                           amplitude=12 * b2.namp)
    # run the LIF model
    (state_monitor, spike_monitor) = simulate_LIF_neuron(input_current=step_current, simulation_time=500 * b2.ms)
    return len(spike_monitor.t)


spike = []
fre = []
for i in tqdm(range(60)):
    for j in tqdm(range(20)):
        f = i + 20
        spike.append(Simulate(f))
        fre.append(f)

plt.xlabel("frequent")
plt.ylabel("spike")
plt.scatter(fre, spike)
plt.savefig("a.pdf")
plt.show()
