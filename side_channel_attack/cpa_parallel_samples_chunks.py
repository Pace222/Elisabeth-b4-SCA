from itertools import product
from time import time

import multiprocessing

from cpa_utils import *

def indices_locations_and_hyps_to_use_for_each_key_nibble_specific_block_idx(key_target_idx: int, block_target_idx: int, total_seeds: np.ndarray, total_traces: np.ndarray, locations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not (0 <= block_target_idx < BLOCK_WIDTH_B4 - 1):
        return None

    key_space = (16,) if block_target_idx % 2 == 0 else (16, 16)
    location_mask_per_trace = np.zeros(total_traces.shape, dtype=bool)
    hypotheses = np.zeros(key_space + total_seeds.shape, dtype=int)

    if block_target_idx % 2 != 0:
        other_key_target_indices = np.zeros_like(total_seeds, dtype=int)

    for i, iv in enumerate(total_seeds):
        indices, whitening = chacha_random_b4(iv)
        keyround_target_idx = indices.index(key_target_idx)
        if keyround_target_idx < KEYROUND_WIDTH_B4:
            round_idx = keyround_target_idx // BLOCK_WIDTH_B4
            block_idx = keyround_target_idx % BLOCK_WIDTH_B4

            # For now, we only attack even-indexed keyrounds (but not the last one) because they depend on a single key nibble
            # TO REMOVE IF BETTER IDEA IS FOUND
            #if block_idx == BLOCK_WIDTH_B4 - 1 or block_idx % 2 != 0:
            #if round_idx != KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4 - 1 or block_idx != 4:
            if (round_idx != KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4 - 1 and round_idx != KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4 - 2) or block_idx != block_target_idx:
                continue
            # TO REMOVE IF BETTER IDEA IS FOUND

            if block_idx != BLOCK_WIDTH_B4 - 1:
                if block_idx % 2 == 0:
                    for k in range(16):
                        block = [0] * BLOCK_WIDTH_B4
                        block[block_idx] = (k + whitening[keyround_target_idx]) % 16
                        sbox_out = s_boxes_b4[block_idx][block[block_idx]]

                        hypotheses[k][i] = HW[sbox_out]
                else:
                    for k1, k2 in product(range(16), range(16)):
                        block = [0] * BLOCK_WIDTH_B4
                        block[block_idx] = (k1 + whitening[keyround_target_idx]) % 16
                        block[block_idx - 1] = (k2 + whitening[keyround_target_idx - 1]) % 16
                        sbox_out = s_boxes_b4[block_idx][(block[block_idx] + block[block_idx - 1]) % 16]
                        
                        hypotheses[k1][k2][i] = HW[sbox_out]
                    other_key_target_indices[i] = indices[keyround_target_idx - 1]
        
                location_mask_per_trace[i][locations[round_idx][block_idx]] = True
            else:
                raise ValueError("Should not happen")
                for i in range(3):
                    block[2*i + 1] = (block[2*i + 1] + block[2*i]) % 16
                y = [s_boxes_b4[i][block[i]] for i in range(BLOCK_WIDTH_B4 - 1)]
                z = [(y[(2*i + 5*j - 1) % (BLOCK_WIDTH_B4 - 1)] + y[2*i + j]) % 16 for i in range(3) for j in range(2)]
                z = [s_boxes_b4[6 + i][(z[i] + block[(i + 2) % (BLOCK_WIDTH_B4 - 1)]) % 16] for i in range(BLOCK_WIDTH_B4 - 1)]
                t_0 = (z[0] + z[1] + z[2]) % 16
                t_0 = (t_0 + block[block_idx - 1]) % 16
                sbox_out = s_boxes_b4[12][t_0]

                hypotheses[i] = HW[(block[block_idx] + sbox_out) % 16]
    
    if block_target_idx % 2 == 0:
        return total_traces[location_mask_per_trace].reshape((-1, locations.shape[2])), hypotheses[..., np.any(location_mask_per_trace, axis=1)]
    else:
        return total_traces[location_mask_per_trace].reshape((-1, locations.shape[2])), hypotheses[..., np.any(location_mask_per_trace, axis=1)], other_key_target_indices
    
def indices_locations_and_hyps_to_use_for_each_key_nibble(key_target_idx: int, block_target_indices: set[int], total_seeds: np.ndarray, total_traces: np.ndarray, locations: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    if any([not (0 <= b < BLOCK_WIDTH_B4 - 1) for b in block_target_indices]):
        return None

    return [indices_locations_and_hyps_to_use_for_each_key_nibble_specific_block_idx(key_target_idx, b, total_seeds, total_traces, locations) for b in block_target_indices]

def main():
    traces_path = "..\\acquisition\\carto_eB4-Rnd-3-WhiteningAndFullFilter-1_key_256000_samples\\carto_eB4-Rnd-3-WhiteningAndFullFilter.mat"
    key_path = "..\\acquisition\\carto_eB4-Rnd-3-WhiteningAndFullFilter-1_key_256000_samples\\carto_eB4-Rnd-3-WhiteningAndFullFilter.log"
    locations_path = "correlation_locations_b4_two_last_rounds.pic"
    seeds, traces, real_keys, correlation_locations = load_data(traces_path, key_path, locations_path)

    reconstructed_keys = np.zeros_like(real_keys)
    for i in range(reconstructed_keys.shape[0]):
        print(f"Key {i}: ", end="")
        visited = set()
        for j in range(reconstructed_keys.shape[1]):
            block_targets = [4]

            start = time()

            cpus = 61
            with multiprocessing.Pool(cpus) as p:
                pool_results = p.starmap(indices_locations_and_hyps_to_use_for_each_key_nibble, zip([j] * cpus, [block_targets] * cpus, np.array_split(seeds[i], cpus), np.array_split(traces[i], cpus), [correlation_locations] * cpus))
            print(time() - start)

            for k, block in enumerate(block_targets):
                if block % 2 == 0:
                    selected_traces, hypotheses = np.concatenate([res[k][0] for res in pool_results]), np.concatenate([res[k][1] for res in pool_results], axis=1)
                else:
                    selected_traces, hypotheses, other_key_indices = np.concatenate([res[k][0] for res in pool_results], axis=1), np.concatenate([res[k][1] for res in pool_results], axis=2), np.concatenate([res[k][2] for res in pool_results]   )

                corrs = np.array([corr_coef_vectorized(hypotheses[l], selected_traces) for l in range(hypotheses.shape[0])])
                #for l in range(len(corrs)):
                #    plt.plot(corrs[l], label=str(l))

                #plt.ylim([-1, 1])
                #plt.legend()
                #plt.show()
                
                max_corrs = np.max(corrs, axis=1)
                best_k = np.argmax(max_corrs)
                reconstructed_keys[i][j] = best_k

                if block % 2 != 0:
                    pass
                    #reconstructed_keys[i][other_key_indices[]]

                print(hex(best_k)[2:].upper(), end="")

                print(max_corrs)
                #plt.plot(max_corrs)
                #plt.xlabel("Key")
                #plt.ylabel("Max correlation across local duration")
                #plt.title(f"Real key nibble: {real_keys[i][j]}")
                #plt.ylim([0, 1])
                #plt.show()

        print()
        print(f"    vs {"".join([hex(k)[2:].upper() for k in real_keys[i]])}")
        print(f"{len(real_keys[i][real_keys[i] != reconstructed_keys[i]])} mistakes on {len(real_keys[i])} nibbles.")
        print()

if __name__ == '__main__':
    main()