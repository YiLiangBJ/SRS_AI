"""
SRS Channel Estimation - Timing and Delay Spread Estimation

This module implements timing and delay spread estimation for SRS (Sounding Reference Signal)
based channel estimation, ported from MATLAB implementation.
"""

import numpy as np
import torch
from scipy.stats import chi2
from typing import Dict, List, Tuple, Any, Optional


def estimate_timing_and_delay_spread(
    simu_params: Dict[str, Any],
    ue_srs_transmit: List[int],
    ue_group_port: Dict[Tuple[int, int], List[int]],
    n_rx: int,
    time_in: np.ndarray,
    num_idft: int,
    is_fft: bool,
    n_cs_loop: np.ndarray,
    ue_idx_loop: np.ndarray,
    port_idx_loop: np.ndarray,
    i_group: int,
    ts: float,
    n_cs_max_srs: int,
    est_offset_all: List[float] = None
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], 
           Dict[int, np.ndarray], Dict[int, np.ndarray], List[float], 
           Dict[int, np.ndarray], Dict[int, Dict[int, np.ndarray]], Dict[int, np.ndarray]]:
    """
    Estimate timing offset and delay spread for SRS channel estimation.
    
    Args:
        simu_params: Simulation parameters dictionary containing UE configurations
        ue_srs_transmit: List of UE indices that are transmitting SRS
        ue_group_port: Dictionary mapping (group, UE) to port lists
        n_rx: Number of receive antennas
        time_in: Time domain received signal [samples, antennas, ...]
        num_idft: Number of IDFT points
        is_fft: Boolean indicating if FFT processing is used
        n_cs_loop: Array of cyclic shift values
        ue_idx_loop: Array of UE indices corresponding to n_cs_loop
        port_idx_loop: Array of port indices corresponding to n_cs_loop
        i_group: Current group index
        ts: Sampling period
        n_cs_max_srs: Maximum cyclic shift value for SRS
        est_offset_all: List to accumulate all estimated offsets
        
    Returns:
        Tuple containing:
        - delay_in_samples_srs: Dictionary of timing offsets per UE and port
        - delay_spread_in_samples_srs: Dictionary of delay spreads per UE and port
        - n_leak_cs: Dictionary of leakage compensation values per UE and port
        - phasor_timing_uncertainty_esti: Dictionary of timing uncertainty phasors per UE
        - n_point_retain_info: Dictionary of retained points info per UE and port
        - est_offset_all: Updated list of all estimated offsets
        - interval: Dictionary of intervals per UE and port
        - pdp_info: Dictionary of PDP information per UE and port
        - energy_sig: Dictionary of signal energy per UE and port
    """
    
    if est_offset_all is None:
        est_offset_all = []
    
    # Initialize output dictionaries
    delay_in_samples_srs = {}
    delay_spread_in_samples_srs = {}
    n_leak_cs = {}
    phasor_timing_uncertainty_esti = {}
    n_point_retain_info = {}
    interval = {}
    pdp_info = {}
    energy_sig = {}
    
    # Calculate intervals for DS and Timing estimation
    for ue_idx in ue_srs_transmit:
        ue_config = simu_params['UE'][ue_idx]
        n_cs_i = ue_config['n_csi_SRS']
        interval[ue_idx] = np.zeros(len(n_cs_i))
        
        for i_cs, n_cs_val in enumerate(n_cs_i):
            idx_all = n_cs_loop == n_cs_val
            temp_cs = np.sort(np.mod(n_cs_max_srs - (n_cs_loop - n_cs_loop[idx_all]), n_cs_max_srs))
            
            if len(temp_cs) > 1:
                interval[ue_idx][i_cs] = temp_cs[1] - temp_cs[0]
                interval[ue_idx][i_cs] = min(interval[ue_idx][i_cs], n_cs_max_srs / 4)
            else:
                interval[ue_idx][i_cs] = n_cs_max_srs / 4
    
    # Determine block size
    tc = 1 / (480e3 * 4096)  # From 38.211 section 4.1
    max_offset_tc = 1856 * tc
    
    if num_idft > 72:
        if num_idft <= 144:
            granularity_factor = 4
        else:
            granularity_factor = 2
        
        blk_size_options_all = np.arange(1, 17)
        flag_mod = np.mod(num_idft / n_cs_max_srs, blk_size_options_all)
        blk_size_options = blk_size_options_all[flag_mod == 0]
        flag = (num_idft / blk_size_options / n_cs_max_srs) >= granularity_factor
        flag_1 = np.where(flag)[0]
        
        if len(flag_1) == 0:
            blk_size = 1
        else:
            blk_size = blk_size_options[flag_1[-1]]
    else:
        blk_size = 1
    
    n_offset_orig = int(np.ceil(max_offset_tc / ts))
    n_offset = int(np.ceil(max_offset_tc / ts / blk_size) * blk_size)
    
    # Number of RX antennas used for timing/delay estimation
    nr_used = simu_params.get('NrUsedforTimingDelayEst', n_rx)
    
    # Initialize dictionaries for intermediate calculations
    timing_range = {}
    max_sample_idx = {}
    est_offset = {}
    time_in_time_uncertainty_power_1d = {}
    next_ue_idx_port_idx = {}
    interval_org = {}
    interval_back = {}
    
    # Estimate timing uncertainty
    for ue_idx in ue_srs_transmit:
        ue_config = simu_params['UE'][ue_idx]
        srs_tx_sequence = ue_config['SRSTXSequence']
        ifft_size_srs = max(32, 2**(int(np.ceil(np.log2(srs_tx_sequence.shape[0])))))
        
        idx_ue = np.where(np.array(ue_srs_transmit) == ue_idx)[0][0]
        ue_group_port_tmp = ue_group_port[(i_group, idx_ue)]
        
        # Initialize arrays for this UE
        timing_range[ue_idx] = np.zeros((len(ue_group_port_tmp), 2))
        max_sample_idx[ue_idx] = np.zeros(len(ue_group_port_tmp))
        est_offset[ue_idx] = np.zeros(len(ue_group_port_tmp))
        delay_in_samples_srs[ue_idx] = np.zeros(len(ue_group_port_tmp))
        time_in_time_uncertainty_power_1d[ue_idx] = {}
        next_ue_idx_port_idx[ue_idx] = {}
        interval_org[ue_idx] = np.zeros(len(ue_group_port_tmp))
        interval_back[ue_idx] = np.zeros(len(ue_group_port_tmp))
        
        for port_idx, i_port in enumerate(ue_group_port_tmp):
            n_cs_i = ue_config['n_csi_SRS'][i_port]
            n_shift = n_cs_i * num_idft / n_cs_max_srs
            n_shift_integer = int(np.floor(n_shift))
            
            if n_shift_integer != n_shift:
                raise ValueError('Wrong n_shift!')
            
            # Refine timing range and limit valid search range
            temp_cs, idx_cs = np.unique(np.mod(n_cs_max_srs - (n_cs_loop - n_cs_i), n_cs_max_srs), return_inverse=True)
            temp_cs = np.sort(temp_cs)
            idx_cs_sorted = np.argsort(np.mod(n_cs_max_srs - (n_cs_loop - n_cs_i), n_cs_max_srs))
            
            if len(idx_cs_sorted) > 1:
                next_ue_idx_port_idx[ue_idx][i_port] = [ue_idx_loop[idx_cs_sorted[1]], port_idx_loop[idx_cs_sorted[1]]]
            else:
                next_ue_idx_port_idx[ue_idx][i_port] = [-1, -1]
            
            temp_cs_back = np.sort(np.mod(n_cs_loop - n_cs_i, n_cs_max_srs))
            
            if len(temp_cs) > 1:
                interval_org[ue_idx][port_idx] = temp_cs[1] - temp_cs[0]
                interval_back[ue_idx][port_idx] = temp_cs_back[1] - temp_cs_back[0]
                interval_org[ue_idx][port_idx] = min(interval_org[ue_idx][port_idx], n_cs_max_srs / 4)
                interval_back[ue_idx][port_idx] = min(interval_back[ue_idx][port_idx], n_cs_max_srs / 4)
            else:
                interval_org[ue_idx][port_idx] = n_cs_max_srs / 4
                interval_back[ue_idx][port_idx] = n_cs_max_srs / 4
            
            # Max timing search spacing in samples
            timing_search_forward_samples = min(
                int(np.floor(interval_org[ue_idx][port_idx] * num_idft / n_cs_max_srs / 2)), 
                n_offset_orig
            )
            timing_search_backward_samples = min(
                int(np.floor(interval_back[ue_idx][port_idx] * num_idft / n_cs_max_srs / 2)), 
                n_offset_orig
            )
            
            # Timing range
            timing_range[ue_idx][port_idx, 0] = n_offset - (timing_search_backward_samples - 1)
            timing_range[ue_idx][port_idx, 1] = n_offset + timing_search_forward_samples - 1
            
            # Apply time domain shift
            time_in_time_uncertainty = np.roll(time_in, n_offset + n_shift_integer, axis=0)
            time_in_time_uncertainty_power = np.abs(time_in_time_uncertainty)**2
            
            # Sum over appropriate RX antennas
            if (n_rx <= nr_used) or (n_rx // nr_used != np.ceil(n_rx / nr_used)):
                time_in_time_uncertainty_power_1d[ue_idx][i_port] = np.sum(time_in_time_uncertainty_power, axis=(1, 2))
                nr_used = n_rx
            else:
                step = n_rx // nr_used
                time_in_time_uncertainty_power_1d[ue_idx][i_port] = np.sum(
                    time_in_time_uncertainty_power[:, :, ::step], axis=(1, 2)
                )
            
            # Prepare timing signal
            time_in_time_uncertainty_power_1d_timing = time_in_time_uncertainty_power_1d[ue_idx][i_port].copy()
            time_in_time_uncertainty_power_1d_timing[:n_offset - n_offset_orig] = 0
            
            # Find time offset index
            start_idx = int(timing_range[ue_idx][port_idx, 0])
            end_idx = int(timing_range[ue_idx][port_idx, 1]) + 1
            search_window = time_in_time_uncertainty_power_1d_timing[start_idx:end_idx]
            max_sample_idx[ue_idx][port_idx] = np.argmax(search_window)
            
            # Calculate estimated offset
            est_offset[ue_idx][port_idx] = -n_offset + (start_idx + max_sample_idx[ue_idx][port_idx] - 1)
            
            if is_fft:
                delay_in_samples_srs[ue_idx][port_idx] = np.ceil(
                    est_offset[ue_idx][port_idx] * ifft_size_srs / num_idft
                )
            else:
                delay_in_samples_srs[ue_idx][port_idx] = est_offset[ue_idx][port_idx]
            
            est_offset_all.append(est_offset[ue_idx][port_idx])
    
    # Estimate delay spread
    for ue_idx in ue_srs_transmit:
        ue_config = simu_params['UE'][ue_idx]
        srs_tx_sequence = ue_config['SRSTXSequence']
        ifft_size_srs = max(32, 2**(int(np.ceil(np.log2(srs_tx_sequence.shape[0])))))
        
        idx_ue = np.where(np.array(ue_srs_transmit) == ue_idx)[0][0]
        ue_group_port_tmp = ue_group_port[(i_group, idx_ue)]
        
        n_point_retain_info[ue_idx] = np.zeros((2, len(ue_group_port_tmp)))
        delay_spread_in_samples_srs[ue_idx] = np.zeros(len(ue_group_port_tmp))
        n_leak_cs[ue_idx] = np.zeros(len(ue_group_port_tmp))
        pdp_info[ue_idx] = {}
        energy_sig[ue_idx] = np.zeros(len(ue_group_port_tmp))
        
        for port_idx, i_port in enumerate(ue_group_port_tmp):
            # Calculate CIR range power with block processing
            signal_power = time_in_time_uncertainty_power_1d[ue_idx][i_port]
            cir_range_power_blk = np.sum(signal_power.reshape(-1, blk_size), axis=1)
            
            # Determine shift for delay spread estimation
            if est_offset[ue_idx][port_idx] > num_idft / n_cs_max_srs / 2:
                n_shift_for_ds = -n_offset + num_idft / n_cs_max_srs / 2
            else:
                n_shift_for_ds = -est_offset[ue_idx][port_idx] - n_offset + num_idft / n_cs_max_srs / 2
            
            n_shift_for_ds_blk = int(np.round(n_shift_for_ds / blk_size))
            
            # Apply circular shift and determine used range
            cir_range_power_blk_used = np.roll(cir_range_power_blk, n_shift_for_ds_blk)
            
            # Calculate next CS timing offset
            if next_ue_idx_port_idx[ue_idx][i_port][0] != -1:
                next_ue_idx = next_ue_idx_port_idx[ue_idx][i_port][0]
                next_port_idx = next_ue_idx_port_idx[ue_idx][i_port][1]
                next_cs_timing_offset_samples = (
                    interval_org[ue_idx][port_idx] * num_idft / n_cs_max_srs +
                    est_offset[next_ue_idx][next_port_idx] - est_offset[ue_idx][port_idx]
                )
            else:
                next_cs_timing_offset_samples = (
                    interval_org[ue_idx][port_idx] * num_idft / n_cs_max_srs - est_offset[ue_idx][port_idx]
                )
            
            next_cs_timing_offset_blk_idx = int(np.ceil(
                (1 + n_offset + est_offset[ue_idx][port_idx] + 
                 n_shift_for_ds_blk * blk_size + next_cs_timing_offset_samples) / blk_size
            ))
            
            max_len = min(
                int(len(cir_range_power_blk_used) / n_cs_max_srs * interval_org[ue_idx][port_idx]),
                next_cs_timing_offset_blk_idx - 1
            )
            cir_range_power_blk_used = cir_range_power_blk_used[:max_len]
            
            # Apply edge detection logic
            if len(cir_range_power_blk_used) > 2:
                if cir_range_power_blk_used[0] > cir_range_power_blk_used[1]:
                    if (est_offset[ue_idx][port_idx] + n_offset + 1 + n_shift_for_ds_blk * blk_size) > blk_size:
                        cir_range_power_blk_used[0] = np.min(cir_range_power_blk_used)
                
                if cir_range_power_blk_used[-1] > cir_range_power_blk_used[-2]:
                    if (est_offset[ue_idx][port_idx] + n_offset + 1 + n_shift_for_ds_blk * blk_size) <= (len(cir_range_power_blk_used) - 1) * blk_size:
                        cir_range_power_blk_used[-1] = np.min(cir_range_power_blk_used)
            
            # Apply threshold detection
            min_value_blk = np.min(cir_range_power_blk_used)
            dof = 2 * blk_size * nr_used
            pfa = 0.001
            tune_factor = 2
            th = tune_factor * chi2.ppf(1 - pfa, dof) / dof
            idx_blk = np.where(cir_range_power_blk_used >= th * min_value_blk)[0] - n_shift_for_ds_blk
            
            # Update signal window with estimated TO and DS
            if len(idx_blk) == 0 or interval_org[ue_idx][port_idx] == 1:
                len_forward = int(np.ceil(2/4 * num_idft / n_cs_max_srs))
                len_backward = int(np.ceil(1/4 * num_idft / n_cs_max_srs))
            else:
                len_forward = idx_blk[-1] * blk_size - (
                    timing_range[ue_idx][port_idx, 0] + max_sample_idx[ue_idx][port_idx] - 1
                ) + 1
                len_backward = (
                    timing_range[ue_idx][port_idx, 0] + max_sample_idx[ue_idx][port_idx] - 1
                ) - ((idx_blk[0] - 1) * blk_size + 1)
                
                len_forward = max(1, int(len_forward))
                len_backward = max(0, int(len_backward))
            
            n_point_retain = len_forward + len_backward
            
            # Convert to FFT size if needed
            if is_fft:
                n_point_retain = int(np.ceil(n_point_retain * ifft_size_srs / num_idft))
                delay_spread_in_samples_srs[ue_idx][port_idx] = n_point_retain
                n_leak_cs[ue_idx][port_idx] = int(np.ceil(len_backward * ifft_size_srs / num_idft))
                n_point_retain_info[ue_idx][:, port_idx] = [len_forward, len_backward]
            else:
                delay_spread_in_samples_srs[ue_idx][port_idx] = n_point_retain
                n_point_retain_info[ue_idx][:, port_idx] = [len_forward, len_backward]
                n_leak_cs[ue_idx][port_idx] = 0
            
            # Calculate PDP and energy
            n_shift_i = int(ue_config['n_csi_SRS'][i_port] * num_idft / n_cs_max_srs)
            time_in_shift = np.roll(time_in, n_shift_i - int(est_offset[ue_idx][port_idx]), axis=0)
            time_in_shift_tailored = time_in_shift.copy()
            
            len_forward_info = int(n_point_retain_info[ue_idx][0, port_idx])
            len_backward_info = int(n_point_retain_info[ue_idx][1, port_idx])
            
            time_in_shift_tailored[len_forward_info:-len_backward_info if len_backward_info > 0 else None, :, :] = 0
            
            # Collect PDP
            pdp_tmp = np.zeros((len_forward_info + len_backward_info, time_in_shift_tailored.shape[1], time_in_shift_tailored.shape[2]))
            if len_backward_info > 0:
                pdp_tmp[:len_backward_info, :, :] = time_in_shift_tailored[-len_backward_info:, :, :]
            pdp_tmp[len_backward_info:len_backward_info + len_forward_info, :, :] = time_in_shift_tailored[:len_forward_info, :, :]
            
            pdp_info[ue_idx][i_port] = np.mean(np.abs(pdp_tmp)**2, axis=(1, 2))
            energy_sig[ue_idx][port_idx] = np.sum(np.mean(np.abs(time_in_shift_tailored)**2, axis=(1, 2)))
    
    # Calculate timing offset phasor for MSE calculation
    for ue_idx in ue_srs_transmit:
        ue_config = simu_params['UE'][ue_idx]
        srs_index_slot = ue_config['SRSIndexSlot']
        scs_srs = np.arange(1, srs_index_slot.shape[0] + 1) - num_idft/2 - 1
        mean_offset = np.mean(est_offset[ue_idx])
        phasor_timing_uncertainty_esti[ue_idx] = np.exp(1j * 2 * np.pi * mean_offset / num_idft * scs_srs)
    
    return (
        delay_in_samples_srs,
        delay_spread_in_samples_srs,
        n_leak_cs,
        phasor_timing_uncertainty_esti,
        n_point_retain_info,
        est_offset_all,
        interval,
        pdp_info,
        energy_sig
    )


def create_simu_params_template() -> Dict[str, Any]:
    """
    Create a template for simulation parameters dictionary.
    
    Returns:
        Template dictionary with expected structure for simu_params
    """
    return {
        'UE': {
            # Example UE configuration
            0: {
                'n_csi_SRS': [0, 1, 2],  # Cyclic shift indices for each port
                'SRSTXSequence': np.random.randn(128, 1) + 1j * np.random.randn(128, 1),  # SRS TX sequence
                'SRSIndexSlot': np.arange(128).reshape(-1, 1)  # SRS subcarrier indices
            }
        },
        'NrUsedforTimingDelayEst': 4  # Number of RX antennas used for estimation
    }


# Example usage and testing function
def test_timing_delay_estimation():
    """
    Test function to demonstrate usage of the timing and delay spread estimation.
    """
    # Create test parameters
    simu_params = create_simu_params_template()
    ue_srs_transmit = [0]
    ue_group_port = {(0, 0): [0, 1, 2]}
    n_rx = 4
    num_samples = 1024
    time_in = np.random.randn(num_samples, n_rx, 1) + 1j * np.random.randn(num_samples, n_rx, 1)
    num_idft = 128
    is_fft = True
    n_cs_loop = np.array([0, 1, 2])
    ue_idx_loop = np.array([0, 0, 0])
    port_idx_loop = np.array([0, 1, 2])
    i_group = 0
    ts = 1e-6  # 1 microsecond sampling period
    n_cs_max_srs = 8
    
    # Run estimation
    results = estimate_timing_and_delay_spread(
        simu_params=simu_params,
        ue_srs_transmit=ue_srs_transmit,
        ue_group_port=ue_group_port,
        n_rx=n_rx,
        time_in=time_in,
        num_idft=num_idft,
        is_fft=is_fft,
        n_cs_loop=n_cs_loop,
        ue_idx_loop=ue_idx_loop,
        port_idx_loop=port_idx_loop,
        i_group=i_group,
        ts=ts,
        n_cs_max_srs=n_cs_max_srs
    )
    
    (delay_in_samples_srs, delay_spread_in_samples_srs, n_leak_cs, 
     phasor_timing_uncertainty_esti, n_point_retain_info, est_offset_all,
     interval, pdp_info, energy_sig) = results
    
    print("Timing and Delay Spread Estimation Results:")
    print(f"Delay in samples: {delay_in_samples_srs}")
    print(f"Delay spread in samples: {delay_spread_in_samples_srs}")
    print(f"Energy per UE/Port: {energy_sig}")
    print(f"All estimated offsets: {est_offset_all}")
    
    return results


if __name__ == "__main__":
    # Run test
    test_timing_delay_estimation()
