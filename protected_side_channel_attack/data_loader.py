import math
import numpy as np
#import cupy as np
import scipy.io as sio

import utils
import log_parser

class EntireTraceIterator:
    def __init__(self, traces_path: str, key_path: str, nr_populations: int, nr_scenarios: int, trace_size: int, traces_per_division: int, parse_output: str):
        self.traces_path = traces_path
        self.key_path = key_path
        self.nr_populations = nr_populations
        self.nr_scenarios = nr_scenarios
        self.trace_size = trace_size
        self.traces_per_division = traces_per_division
        self.parse_output = parse_output

        self.inputs_outputs, self.empty_traces = log_parser.parse(self.key_path)
        self.step = (self.nr_populations * self.nr_scenarios) * (traces_per_division // (self.nr_populations * self.nr_scenarios))
        self.nr_traces = len(self.inputs_outputs) * self.nr_scenarios // (self.nr_scenarios + 1)
        self.len = math.ceil((self.nr_traces - len(self.empty_traces)) / self.step)
        self.curr = 0
        self.start_log = 0

    def __len__(self):
        return self.len

    def __call__(self, target_pop, target_scenario, return_traces=True):
        self.target_pop = target_pop
        self.target_scenario = target_scenario
        self.return_traces = return_traces
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

        if self.return_traces:
            traces_dict = sio.loadmat(self.traces_path, variable_names=[f"data_{t + p * (self.nr_scenarios) + s}" for t in range(start_traces + 1, stop_traces + 1, self.nr_populations * self.nr_scenarios) for p in self.target_pop for s in self.target_scenario])
        inputs_outputs = self.inputs_outputs[self.start_log:stop_log]

        if self.return_traces:
            traces     = [[[] for ss in range(len(self.target_scenario))] for pp in range(len(self.target_pop))]
        seeds      = [[[] for ss in range(len(self.target_scenario))] for pp in range(len(self.target_pop))]
        key        = [[int(c, 16) for c in inputs_outputs[1][0][0]]]
        parsed_output = [[[] for ss in range(len(self.target_scenario))] for pp in range(len(self.target_pop))]
        
        for pp, p in enumerate(self.target_pop):
            for ss, s in enumerate(self.target_scenario):
                for t, io in zip(range(start_traces + 1, stop_traces + 1, self.nr_populations * self.nr_scenarios), range(0, len(inputs_outputs), self.nr_populations * (self.nr_scenarios + 1))):
                    if t + p * (self.nr_scenarios) + s not in self.empty_traces:
                        seeds[pp][ss].append(inputs_outputs[io + p * (self.nr_scenarios + 1)][0][0])
                        if self.return_traces:
                            traces[pp][ss].append(traces_dict[f"data_{t + p * (self.nr_scenarios) + s}"][0, 0][4][:, 0])

                        parsed = []
                        if "keyshares" in self.parse_output:
                            output = inputs_outputs[io + p * (self.nr_scenarios + 1) + s + 1][1]
                            keyshare_str = output.split('|')[0]
                            parsed.append(np.array([[int(keyshare_str[i + j], 16) for j in range(utils.NR_SHARES)] for i in range(0, len(keyshare_str), 2)]))
                        if "perms" in self.parse_output:
                            output = inputs_outputs[io + p * (self.nr_scenarios + 1) + s + 1][1]
                            perms_str = output.split('|')[1]
                            parsed.append(np.array([int(perms_str[i:i+2], 16) for i in range(0, len(perms_str), 2)]))
                        parsed_output[pp][ss].append(parsed)

                seeds[pp][ss] = np.array(seeds[pp][ss])
                if self.return_traces:
                    traces[pp][ss] = np.stack(traces[pp][ss])
                    assert traces[pp][ss].shape[0] == seeds[pp][ss].shape[0]
                    assert traces[pp][ss].shape[1] == self.trace_size
                parsed_output[pp][ss] = [np.stack([parsed_output[pp][ss][oo][o] for oo in range(len(parsed_output[pp][ss]))]) for o in range(len(parsed_output[pp][ss][0]))]
        
                if self.parse_output == "keyshares":
                    assert parsed_output[pp][ss][0].shape == (seeds[pp][ss].shape[0], utils.KEY_WIDTH_B4, utils.NR_SHARES)
                elif self.parse_output == "perms":
                    assert parsed_output[pp][ss][0].shape == (seeds[pp][ss].shape[0], utils.NR_PERMS)
                elif self.parse_output == "keyshares+perms":
                    assert parsed_output[pp][ss][0].shape == (seeds[pp][ss].shape[0], utils.KEY_WIDTH_B4, utils.NR_SHARES)
                    assert parsed_output[pp][ss][1].shape == (seeds[pp][ss].shape[0], utils.NR_PERMS)

        self.curr += 1
        self.start_log += (stop_log - self.start_log)

        if self.return_traces:
            return seeds, traces, key, parsed_output
        else:
            return seeds, key, parsed_output

    def full(self, dataset_size: int):
        if self.target_pop is None or self.target_scenario is None or len(self.target_pop) != 1 or len(self.target_scenario) != 1:
            raise ValueError
        
        seeds, traces = np.zeros((dataset_size,), dtype="<U32"), np.zeros((dataset_size, self.trace_size), dtype=np.int16)
        if self.parse_output == "keyshares":
            parse_output = [np.zeros((dataset_size, utils.KEY_WIDTH_B4, utils.NR_SHARES), dtype=np.int8)]
        elif self.parse_output == "perms":
            parse_output = [np.zeros((dataset_size, utils.NR_PERMS), dtype=np.int8)]
        elif self.parse_output == "keyshares+perms":
            parse_output = [np.zeros((dataset_size, utils.KEY_WIDTH_B4, utils.NR_SHARES), dtype=np.int8), np.zeros((dataset_size, utils.NR_PERMS), dtype=np.int8)]

        for j, (seeds_sub, traces_sub, key, output_sub) in enumerate(self):
            seeds[j * self.traces_per_division // self.nr_populations // self.nr_scenarios:j * self.traces_per_division // self.nr_populations // self.nr_scenarios + seeds_sub[self.target_pop[0]][self.target_scenario[0]].shape[0]] = seeds_sub[self.target_pop[0]][self.target_scenario[0]]
            traces[j * self.traces_per_division // self.nr_populations // self.nr_scenarios:j * self.traces_per_division // self.nr_populations // self.nr_scenarios + traces_sub[self.target_pop[0]][self.target_scenario[0]].shape[0]] = traces_sub[self.target_pop[0]][self.target_scenario[0]]
            if self.parse_output:
                for p, parsed in enumerate(output_sub[self.target_pop[0]][self.target_scenario[0]]):
                    parse_output[p][j * self.traces_per_division // self.nr_populations // self.nr_scenarios:j * self.traces_per_division // self.nr_populations // self.nr_scenarios + len(parsed)] = parsed

        seeds = seeds[np.any(traces > 0, axis=1)]
        if self.parse_output:
            for p in range(len(parse_output)):
                parse_output[p] = parse_output[p][np.any(traces > 0, axis=1)]
        traces = traces[np.any(traces > 0, axis=1)]

        return seeds, traces, np.array(key[0]), parse_output
