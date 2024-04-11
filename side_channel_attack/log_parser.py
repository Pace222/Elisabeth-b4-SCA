from typing import List, Tuple, Optional

import re

DELIMITER = ','

regex_input = r"Command sent    = <(4|B4),(\d+),([a-zA-Z0-9]+)((?:,[a-fA-F0-9]+)*)>$"
regex_output = r"Card's response = ((?:[a-fA-F0-9]{2})*) / $"

def parse(log_path: str, actual_args_only: bool = True) -> List[Optional[Tuple[Tuple[str], str]]]:
    """
        Opens the log under log_path and returns a list of input/output pairs. If a pair is None, it means the trace is invalid.
        - An input is a list of strings, representing each passed argument. If actual_args_only is True, only the randomized arguments are returned, not the cipher type, the number of repetitions or the chosen scenario.
        - An output is a the string returned by the Arduino.
    """
    inps_outs = []
    empty_indices = []
    inp = out = None
    ready = False
    with open(log_path, 'r') as log:
        for line in log.readlines():
            if 'WARNING' in line:
                empty_indices.append(int(line[line.index("Acq ") + len("Acq "):line.index("WARNING") - 2]))
            elif 'INFO' in line:
                ready = True
                inp = out = None
            elif ready:
                if inp:
                    out = re.search(regex_output, line)
                    if out:
                        out = out.group(1)
                        out = bytearray.fromhex(out).decode("ASCII")
                        if "Format" in out:
                            inps_outs.append(None)
                            ready = False
                            inp = out = None
                        else:
                            out = out.split(DELIMITER)
                            if not all([out[i] == out[0] for i in range(len(out))]) or not all ([re.search(r"^[a-fA-F0-9]+$", o) for o in out[0]]):
                                inps_outs.append(None)
                                ready = False
                                inp = out = None
                            else:
                                inps_outs.append((inp if not actual_args_only else inp[3:], out[0]))
                                ready = False
                                inp = out = None
                    else:
                        inps_outs.append(None)
                        ready = False
                        inp = out = None
                else:
                    inp = re.search(regex_input, line)
                    if inp:
                        inp = inp.groups()
                        if len(inp) < 3:
                            inps_outs.append(None)
                            ready = False
                            inp = out = None
                        else:
                            inp = inp[:-1] + tuple(inp[-1].split(DELIMITER)[1:])
                            out = None
                    else:
                        inps_outs.append(None)
                        ready = False
                        inp = out = None
            else:
                inp = out = None

    return inps_outs, empty_indices