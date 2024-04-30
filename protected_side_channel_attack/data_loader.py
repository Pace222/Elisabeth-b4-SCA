from typing import Tuple, Union
import pickle as pic

import math
import numpy as np
#import cupy as np
import scipy.io as sio

from utils import KEY_WIDTH_B4, NR_SHARES
import log_parser

class EntireTraceIterator:
    def __init__(self, traces_path: str, key_path: str, nr_populations: int, nr_scenarios: int, trace_size: int = 850_000, traces_per_division: int = 50_000):
        self.traces_path = traces_path
        self.key_path = key_path
        self.nr_populations = nr_populations
        self.nr_scenarios = nr_scenarios
        self.trace_size = trace_size

        self.inputs_outputs, self.empty_traces = log_parser.parse(self.key_path)
        self.step = (self.nr_populations * self.nr_scenarios) * (traces_per_division // (self.nr_populations * self.nr_scenarios))
        self.nr_traces = len(self.inputs_outputs) * self.nr_scenarios // (self.nr_scenarios + 1)
        self.len = math.ceil(self.nr_traces / self.step)
        self.curr = 0
        self.start_log = 0

    def __len__(self):
        return self.len

    def __call__(self, target_pop, target_scenario):
        self.target_pop = target_pop
        self.target_scenario = target_scenario
        return self

    def __iter__(self):
        self.curr = 0
        self.start_log = 0
        return self

    def __next__(self):
        if self.curr >= self.len:
            raise StopIteration
        
        start_traces = self.curr * self.step
        stop_traces  = min(self.nr_traces, (self.curr + 1) * self.step)
        nr_selected_traces = stop_traces - start_traces
        assert nr_selected_traces % (self.nr_populations * self.nr_scenarios) == 0
        stop_log     = self.start_log + nr_selected_traces + nr_selected_traces // self.nr_scenarios

        traces_dict = sio.loadmat(self.traces_path, variable_names=[f"data_{t + p * (self.nr_scenarios) + s}" for t in range(start_traces + 1, stop_traces + 1, self.nr_populations * self.nr_scenarios) for p in self.target_pop for s in self.target_scenario])
        inputs_outputs = self.inputs_outputs[self.start_log:stop_log]

        traces     = [[[] for s in range(len(self.target_scenario))] for p in range(len(self.target_pop))]
        seeds      = [[[] for s in range(len(self.target_scenario))] for p in range(len(self.target_pop))]
        key        = [[int(c, 16) for c in inputs_outputs[1][0][0]]]
        key_shares = [[[] for s in range(len(self.target_scenario))] for p in range(len(self.target_pop))]
        
        for pp, p in enumerate(self.target_pop):
            for ss, s in enumerate(self.target_scenario):
                for t, io in zip(range(start_traces + 1, stop_traces + 1, self.nr_populations * self.nr_scenarios), range(0, len(inputs_outputs), self.nr_populations * (self.nr_scenarios + 1))):
                    if t + p * (self.nr_scenarios) + s not in self.empty_traces:
                        seeds[pp][ss].append(inputs_outputs[io + p * (self.nr_scenarios + 1)][0][0])
                        traces[pp][ss].append(traces_dict[f"data_{t + p * (self.nr_scenarios) + s}"][0, 0][4][:, 0])

                        output = inputs_outputs[io + p * (self.nr_scenarios + 1) + s + 1][1]
                        #keyshare_str = "".join([chr(int(output[i:i+2], 16)) for i in range(0, len(output), 2)]).split('|')[0]
                        #key_shares[pp][ss].append(np.array([[int(keyshare_str[i], 16), int(keyshare_str[i + 1], 16)] for i in range(0, len(keyshare_str), 2)]))
                        
                seeds[pp][ss] = np.array(seeds[pp][ss])
                traces[pp][ss] = np.stack(traces[pp][ss])
                assert traces[pp][ss].shape[1] == self.trace_size
                #key_shares[pp][ss] = np.array(key_shares[pp][ss])
        
                assert seeds[pp][ss].shape[0] == traces[pp][ss].shape[0]
                #assert key_shares[pp][ss].shape == (KEY_WIDTH_B4, NR_SHARES)
                
        self.curr += 1
        self.start_log += (stop_log - self.start_log)

        return seeds, traces, key, key_shares
