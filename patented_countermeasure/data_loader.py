import math
import numpy as np
import scipy.io as sio

from cpa_utils import chacha_random_b4

import utils
import log_parser

class EntireTraceIterator:
    def __init__(self, traces_path: str, key_path: str, trace_size: int, traces_per_division: int, parse_output: str):
        self.traces_path = traces_path
        self.key_path = key_path
        self.trace_size = trace_size
        self.traces_per_division = traces_per_division
        self.parse_output = parse_output

        self.inputs_outputs, self.empty_traces = log_parser.parse(self.key_path)
        self.step = 6 * (traces_per_division // 6)
        self.nr_traces = len(self.inputs_outputs) * 6 // 11
        self.len = math.ceil((self.nr_traces - len(self.empty_traces)) / self.step)
        self.curr = 0
        self.start_log = 0
        
        self.return_traces = True

    def __len__(self):
        return self.len

    def __call__(self, return_traces=True):
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
        assert nr_selected_traces % 6 == 0
        stop_log     = self.start_log + 11 * nr_selected_traces // 6

        if self.return_traces:
            traces_dict = sio.loadmat(self.traces_path, variable_names=[f"data_{t + i}" for t in range(start_traces + 1, stop_traces + 1, 6) for i in range(6)])
        inputs_outputs = self.inputs_outputs[self.start_log:stop_log]

        if self.return_traces:
            traces    = [[], [], [], [], [], []]
        seeds         = [[], [], [], [], [], []]
        labels        = [[], [], [], [], [], []]
        keys          = np.array([[int(c, 16) for c in inputs_outputs[2][0][0]], [int(c, 16) for c in inputs_outputs[3][0][0]], 
                                  [int(c, 16) for c in inputs_outputs[4][0][0]], [int(c, 16) for c in inputs_outputs[6][0][0]],
                                  [int(c, 16) for c in inputs_outputs[8][0][0]], [int(c, 16) for c in inputs_outputs[10][0][0]]])
        parsed_output = [[], [], [], [], [], []]

        for t, io in zip(range(start_traces + 1, stop_traces + 1, 6), range(0, len(inputs_outputs), 11)):
            SEEDS_I  = (0, 0, 0, 0, 0,  0)
            TRACES_I = (0, 1, 2, 3, 4,  5)
            OUTPUT_I = (2, 3, 4, 6, 8, 10)
            seeds_i  = []
            traces_i = []
            output_i = []
            for i in range(6):
                if t + i not in self.empty_traces:
                    seeds_i.append((i, SEEDS_I[i]))
                    traces_i.append((i, TRACES_I[i]))
                    output_i.append((i, OUTPUT_I[i]))


            for i, s in seeds_i:
                seed = inputs_outputs[io + s][0][0]
                seeds[i].append(seed)
                ind, whi = chacha_random_b4(seed)
                labels[i].append((keys[i, ind[:utils.KEYROUND_WIDTH_B4]] + whi) % 16)

            if self.return_traces:
                for i, tt in traces_i:
                    traces[i].append(traces_dict[f"data_{t + tt}"][0, 0][4][:, 0])
            
            if "delays" in self.parse_output:
                for i, o in output_i:
                    output = inputs_outputs[io + o][1]
                    delays_str = output.split(')')[0][1:]
                    parsed_output[i].append(np.array([int(d, 16) for d in delays_str.split(';')]))

        for i in range(6):
            seeds[i] = np.array(seeds[i])
            labels[i] = np.array(labels[i])
            if self.return_traces:
                traces[i] = np.stack(traces[i])
                assert traces[i].shape[0] == seeds[i].shape[0]
                assert traces[i].shape[1] == self.trace_size

            parsed_output[i] = np.stack(parsed_output[i])

            if self.parse_output == "delays":
                assert parsed_output[i].shape == (seeds[i].shape[0], 3 * utils.KEYROUND_WIDTH_B4)

        self.curr += 1
        self.start_log += (stop_log - self.start_log)

        if self.return_traces:
            return seeds, labels, traces, keys, parsed_output
        else:
            return seeds, labels, keys, parsed_output

    def full(self, dataset_size: int):        
        seeds  = np.zeros((6, dataset_size // 6,), dtype="<U32")
        labels = np.zeros((6, dataset_size // 6, utils.KEYROUND_WIDTH_B4), dtype=int)
        traces = np.zeros((6, dataset_size // 6, self.trace_size), dtype=np.int16)
        if self.parse_output == "delays":
            parse_output = np.zeros((6, dataset_size // 6, 3 * utils.KEYROUND_WIDTH_B4))
        
        for j, (seeds_sub, labels_sub, traces_sub, keys, output_sub) in enumerate(self):
            for i in range(6):
                seeds[i, j * self.traces_per_division // 6:j * self.traces_per_division // 6 + seeds_sub[i].shape[0]] = seeds_sub[i]
                labels[i, j * self.traces_per_division // 6:j * self.traces_per_division // 6 + labels_sub[i].shape[0]] = labels_sub[i]
                traces[i, j * self.traces_per_division // 6:j * self.traces_per_division // 6 + traces_sub[i].shape[0]] = traces_sub[i]
                if self.parse_output:
                    parse_output[i, j * self.traces_per_division // 6:j * self.traces_per_division // 6 + output_sub[i].shape[0]] = output_sub[i]

        seeds = [seeds[i, np.any(traces[i] > 0, axis=1)] for i in range(6)]
        labels = [labels[i, np.any(traces[i] > 0, axis=1)] for i in range(6)]
        if self.parse_output:
            parse_output = [parse_output[i, np.any(traces[i] > 0, axis=1)] for i in range(6)]
        traces = [traces[i, np.any(traces[i] > 0, axis=1)] for i in range(6)]

        return *seeds, *labels, *traces, *keys, *parse_output
