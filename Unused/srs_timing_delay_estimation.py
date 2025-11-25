"""
SRS Timing and Delay Spread Estimation

This module implements timing and delay spread estimation for SRS (Sounding Reference Signal)
based on LS (Least Squares) channel estimation in time domain, ported from MATLAB.

Author: AI Assistant
Date: 2025-07-21
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import math


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
    est_offset_all: List[float]
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], 
           Dict[int, np.ndarray], Dict[int, np.ndarray], List[float], 
           Dict[int, np.ndarray], Dict[int, Dict[int, np.ndarray]], Dict[int, np.ndarray]]:
    """
    Estimate timing and delay spread for SRS channels using LS estimation in time domain.
    
    Args:
        simu_params: Dictionary containing simulation parameters including UE configurations
        ue_srs_transmit: List of UE indices that are transmitting SRS
        ue_group_port: Dictionary mapping (group, UE) to port lists
        n_rx: Number of receive antennas
        time_in: Time domain input signal [samples, antennas, slots]
        num_idft: Number of IDFT points
        is_fft: Boolean indicating if FFT processing is used
        n_cs_loop: Array of cyclic shift values in the loop
        ue_idx_loop: Array of UE indices in the loop
        port_idx_loop: Array of port indices in the loop
        i_group: Current group index
        ts: Sampling period
        n_cs_max_srs: Maximum cyclic shift value for SRS
        est_offset_all: List to accumulate all estimated offsets
    
    Returns:
        Tuple containing:
        - delay_in_samples_srs: Dictionary of timing delays per UE and port
        - delay_spread_in_samples_srs: Dictionary of delay spreads per UE and port
        - n_leak_cs: Dictionary of cyclic shift leakage values
        - phasor_timing_uncertainty_esti: Dictionary of timing uncertainty phasors
        - n_point_retain_info: Dictionary of retained point information
        - est_offset_all: Updated list of all estimated offsets
        - interval: Dictionary of intervals for each UE and port
        - pdp_info: Dictionary of Power Delay Profile information
        - energy_sig: Dictionary of signal energy per UE and port
    """
    
    # Initialize output dictionaries
    delay_in_samples_srs = {}
    delay_spread_in_samples_srs = {}
    n_leak_cs = {}
    phasor_timing_uncertainty_esti = {}
    n_point_retain_info = {}
    interval = {}
    interval_org = {}
    interval_back = {}
    pdp_info = {}
    energy_sig = {}
    est_offset = {}
    timing_range = {}
    next_ue_idx_port_idx = {}
    time_in_time_uncertainty_power_1d = {}
    max_sample_idx = {}
    
    # Calculate the interval for DS and Timing estimation
    for ue_idx in ue_srs_transmit:
        n_cs_i = simu_params['UE'][ue_idx]['n_csi_SRS']
        interval[ue_idx] = np.zeros(len(n_cs_i))
        
        for i_cs in range(len(n_cs_i)):
            idx_all = n_cs_i[i_cs] == n_cs_loop
            if np.any(idx_all):
                cs_value = n_cs_i[i_cs]  # Use the actual CS value directly
                temp_cs = np.sort(np.mod(n_cs_max_srs - (n_cs_loop - cs_value), n_cs_max_srs))
            else:
                temp_cs = np.array([])
            
            if len(temp_cs) > 1:
                interval[ue_idx][i_cs] = temp_cs[1] - temp_cs[0]
                interval[ue_idx][i_cs] = min(interval[ue_idx][i_cs], n_cs_max_srs / 4)
            else:
                interval[ue_idx][i_cs] = n_cs_max_srs / 4
    
    # Determine the block size
    # Calculate the maximum advance/delay value in the current configuration
    # Tc is defined in section 4.1 of 38.211
    tc = 1 / (480e3 * 4096)  # 480kHz, 4096 FFT points
    max_offset_tc = 1856 * tc  # 640*Tc is the maximum offset specified by the standard
    
    if num_idft > 72:
        # Logic to decide BlkSize and nBlk
        if num_idft <= 144:
            granularity_factor = 4
        else:
            granularity_factor = 2
        
        blk_size_options_all = np.arange(1, 17)
        flag_mod = np.mod(num_idft / n_cs_max_srs, blk_size_options_all)
        blk_size_options = blk_size_options_all[flag_mod == 0]
        flag = (num_idft / blk_size_options / n_cs_max_srs) >= granularity_factor
        flag_1 = np.where(flag == 1)[0]
        
        if len(flag_1) == 0:
            blk_size = 1
        else:
            blk_size = blk_size_options[flag_1[-1]]
    else:
        blk_size = 1
    
    n_offset_orig = math.ceil(max_offset_tc / ts)
    n_offset = math.ceil(max_offset_tc / ts / blk_size) * blk_size
    
    # Number of receive antennas used for timing/delay estimation
    nr_used = simu_params.get('NrUsedforTimingDelayEst', n_rx)
    
    # Estimate the timing uncertainty
    for ue_idx in ue_srs_transmit:
        srs_tx_sequence_size = simu_params['UE'][ue_idx]['SRSTXSequence'].shape[0]
        ifft_size_srs = max(32, 2**(math.ceil(math.log2(srs_tx_sequence_size))))
        idx_ue = np.where(np.array(ue_srs_transmit) == ue_idx)[0][0]
        ue_group_port_tmp = ue_group_port[(i_group, idx_ue)]
        
        # Initialize dictionaries for this UE
        est_offset[ue_idx] = np.zeros(len(ue_group_port_tmp))
        delay_in_samples_srs[ue_idx] = np.zeros(len(ue_group_port_tmp))
        max_sample_idx[ue_idx] = np.zeros(len(ue_group_port_tmp))
        timing_range[ue_idx] = np.zeros((len(ue_group_port_tmp), 2))
        next_ue_idx_port_idx[ue_idx] = {}
        time_in_time_uncertainty_power_1d[ue_idx] = {}
        interval_org[ue_idx] = np.zeros(len(ue_group_port_tmp))
        interval_back[ue_idx] = np.zeros(len(ue_group_port_tmp))
        
        for port_idx, i_port in enumerate(ue_group_port_tmp):
            n_cs_i = simu_params['UE'][ue_idx]['n_csi_SRS'][i_port]
            n_shift = n_cs_i * num_idft / n_cs_max_srs
            n_shift_integer = int(np.floor(n_shift))
            
            if n_shift_integer != n_shift:
                raise ValueError('Wrong n_shift!')
            
            # Refine the TimingRange and limit the valid search range
            temp_cs, idx_cs = np.unique(np.mod(n_cs_max_srs - (n_cs_loop - n_cs_i), n_cs_max_srs), 
                                       return_index=True)
            temp_cs = np.sort(temp_cs)
            idx_cs = np.argsort(np.mod(n_cs_max_srs - (n_cs_loop - n_cs_i), n_cs_max_srs))
            
            if len(idx_cs) > 1:
                next_ue_idx_port_idx[ue_idx][i_port] = [ue_idx_loop[idx_cs[1]], port_idx_loop[idx_cs[1]]]
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
            
            # Max Timing Search Spacing in Samples
            timing_search_forward_samples = min(
                int(np.floor(interval_org[ue_idx][port_idx] * num_idft / n_cs_max_srs / 2)),
                n_offset_orig
            )
            timing_search_backward_samples = min(
                int(np.floor(interval_back[ue_idx][port_idx] * num_idft / n_cs_max_srs / 2)),
                n_offset_orig
            )
            
            # After time domain shift, the range of interest
            timing_range[ue_idx][port_idx, 0] = n_offset - (timing_search_backward_samples - 1)  # start index
            timing_range[ue_idx][port_idx, 1] = n_offset + timing_search_forward_samples - 1     # end index
            
            # Apply circular shift to time domain signal
            time_in_time_uncertainty = np.roll(time_in, n_offset + n_shift_integer, axis=0)
            time_in_time_uncertainty_power = np.abs(time_in_time_uncertainty)**2
            
            # Sum across antennas and slots based on Nr_used
            if (n_rx <= nr_used) or (n_rx // nr_used != math.ceil(n_rx / nr_used)):
                time_in_time_uncertainty_power_1d[ue_idx][i_port] = np.sum(
                    time_in_time_uncertainty_power, axis=(1, 2)
                )
                nr_used = n_rx
            else:
                antenna_step = n_rx // nr_used
                time_in_time_uncertainty_power_1d[ue_idx][i_port] = np.sum(
                    time_in_time_uncertainty_power[:, :, ::antenna_step], axis=(1, 2)
                )
            
            # Prepare for timing estimation
            time_in_time_uncertainty_power_1d_timing = time_in_time_uncertainty_power_1d[ue_idx][i_port].copy()
            time_in_time_uncertainty_power_1d_timing[:n_offset - n_offset_orig] = 0
            
            # Find the time offset index
            start_idx = int(timing_range[ue_idx][port_idx, 0])
            end_idx = int(timing_range[ue_idx][port_idx, 1])
            
            # Ensure valid search window
            if start_idx >= len(time_in_time_uncertainty_power_1d_timing) or end_idx < start_idx:
                # Default to center if search window is invalid
                max_sample_idx[ue_idx][port_idx] = 0
                est_offset[ue_idx][port_idx] = 0
            else:
                # Clip indices to valid range
                start_idx = max(0, min(start_idx, len(time_in_time_uncertainty_power_1d_timing) - 1))
                end_idx = max(start_idx, min(end_idx, len(time_in_time_uncertainty_power_1d_timing) - 1))
                
                search_window = time_in_time_uncertainty_power_1d_timing[start_idx:end_idx+1]
                if len(search_window) > 0:
                    max_sample_idx[ue_idx][port_idx] = np.argmax(search_window)
                    est_offset[ue_idx][port_idx] = -n_offset + (start_idx + max_sample_idx[ue_idx][port_idx])
                else:
                    max_sample_idx[ue_idx][port_idx] = 0
                    est_offset[ue_idx][port_idx] = 0
            
            if is_fft:
                delay_in_samples_srs[ue_idx][port_idx] = math.ceil(
                    est_offset[ue_idx][port_idx] * ifft_size_srs / num_idft
                )
            else:
                delay_in_samples_srs[ue_idx][port_idx] = est_offset[ue_idx][port_idx]
            
            est_offset_all.append(est_offset[ue_idx][port_idx])
    
    # Estimate the delay spread
    for ue_idx in ue_srs_transmit:
        srs_tx_sequence_size = simu_params['UE'][ue_idx]['SRSTXSequence'].shape[0]
        ifft_size_srs = max(32, 2**(math.ceil(math.log2(srs_tx_sequence_size))))
        idx_ue = np.where(np.array(ue_srs_transmit) == ue_idx)[0][0]
        ue_group_port_tmp = ue_group_port[(i_group, idx_ue)]
        
        # Initialize arrays for this UE
        n_point_retain_info[ue_idx] = np.zeros((2, len(ue_group_port_tmp)))
        delay_spread_in_samples_srs[ue_idx] = np.zeros(len(ue_group_port_tmp))
        n_leak_cs[ue_idx] = np.zeros(len(ue_group_port_tmp))
        pdp_info[ue_idx] = {}
        energy_sig[ue_idx] = np.zeros(len(ue_group_port_tmp))
        
        for port_idx, i_port in enumerate(ue_group_port_tmp):
            # Calculate block-wise power
            power_1d = time_in_time_uncertainty_power_1d[ue_idx][i_port]
            cir_range_power_blk = np.sum(power_1d.reshape(-1, blk_size), axis=1)
            
            # Determine shift for delay spread estimation
            if est_offset[ue_idx][port_idx] > num_idft / n_cs_max_srs / 2:
                n_shift_for_ds = -n_offset + num_idft / n_cs_max_srs / 2
            else:
                n_shift_for_ds = -est_offset[ue_idx][port_idx] - n_offset + num_idft / n_cs_max_srs / 2
            
            n_shift_for_ds_blk = round(n_shift_for_ds / blk_size)
            
            # Apply circular shift to block power
            cir_range_power_blk_used = np.roll(cir_range_power_blk, n_shift_for_ds_blk)
            
            # Calculate next CS timing offset
            if next_ue_idx_port_idx[ue_idx][i_port][0] != -1:
                next_ue = next_ue_idx_port_idx[ue_idx][i_port][0]
                next_port = next_ue_idx_port_idx[ue_idx][i_port][1]
                next_cs_timing_offset_samples = (
                    interval_org[ue_idx][port_idx] * num_idft / n_cs_max_srs +
                    est_offset[next_ue][next_port] - est_offset[ue_idx][port_idx]
                )
            else:
                next_cs_timing_offset_samples = (
                    interval_org[ue_idx][port_idx] * num_idft / n_cs_max_srs - 
                    est_offset[ue_idx][port_idx]
                )
            
            next_cs_timing_offset_blk_idx = math.ceil(
                (1 + n_offset + est_offset[ue_idx][port_idx] + n_shift_for_ds_blk * blk_size + 
                 next_cs_timing_offset_samples) / blk_size
            )
            
            # Limit the range of CIR power
            max_len = min(len(cir_range_power_blk_used) // n_cs_max_srs * interval_org[ue_idx][port_idx],
                         next_cs_timing_offset_blk_idx - 1)
            cir_range_power_blk_used = cir_range_power_blk_used[:int(max_len)]
            
            # Handle edge cases for first and last blocks
            if len(cir_range_power_blk_used) > 2:
                if cir_range_power_blk_used[0] > cir_range_power_blk_used[1]:
                    if (est_offset[ue_idx][port_idx] + n_offset + 1 + n_shift_for_ds_blk * blk_size) > blk_size:
                        cir_range_power_blk_used[0] = np.min(cir_range_power_blk_used)
                
                if cir_range_power_blk_used[-1] > cir_range_power_blk_used[-2]:
                    if (est_offset[ue_idx][port_idx] + n_offset + 1 + n_shift_for_ds_blk * blk_size) <= (len(cir_range_power_blk_used) - 1) * blk_size:
                        cir_range_power_blk_used[-1] = np.min(cir_range_power_blk_used)
            
            # Determine threshold using chi-squared distribution
            min_value_blk = np.min(cir_range_power_blk_used)
            dof = 2 * blk_size * nr_used
            pfa = 0.001
            tune_factor = 2
            th = tune_factor * stats.chi2.ppf(1 - pfa, dof) / dof
            
            idx_blk = np.where(cir_range_power_blk_used >= th * min_value_blk)[0] - n_shift_for_ds_blk
            
            # Update the signal window with estimated TO and DS
            if len(idx_blk) == 0 or interval_org[ue_idx][port_idx] == 1:
                len_forward = math.ceil(2/4 * num_idft / n_cs_max_srs)
                len_backward = math.ceil(1/4 * num_idft / n_cs_max_srs)
            else:
                timing_peak_idx = int(timing_range[ue_idx][port_idx, 0] + max_sample_idx[ue_idx][port_idx])
                len_forward = idx_blk[-1] * blk_size - timing_peak_idx + 1
                len_backward = timing_peak_idx - (idx_blk[0] - 1) * blk_size - 1
                
                len_forward = max(1, len_forward)
                len_backward = max(0, len_backward)
            
            n_point_retain = len_forward + len_backward
            
            # Scale to IFFT size if needed
            if is_fft:
                n_point_retain = math.ceil(n_point_retain * ifft_size_srs / num_idft)
                delay_spread_in_samples_srs[ue_idx][port_idx] = n_point_retain
                n_leak_cs[ue_idx][port_idx] = math.ceil(len_backward * ifft_size_srs / num_idft)
                n_point_retain_info[ue_idx][:, port_idx] = [len_forward, len_backward]
            else:
                delay_spread_in_samples_srs[ue_idx][port_idx] = n_point_retain
                n_point_retain_info[ue_idx][:, port_idx] = [len_forward, len_backward]
                n_leak_cs[ue_idx][port_idx] = 0
            
            # Calculate PDP and energy information
            n_cs_i = simu_params['UE'][ue_idx]['n_csi_SRS'][i_port]
            n_shift_integer = int(n_cs_i * num_idft / n_cs_max_srs)
            
            time_in_shift = np.roll(time_in, n_shift_integer - int(est_offset[ue_idx][port_idx]), axis=0)
            time_in_shift_tailored = time_in_shift.copy()
            
            # Zero out parts outside the retained window
            len_forward_int = int(n_point_retain_info[ue_idx][0, port_idx])
            len_backward_int = int(n_point_retain_info[ue_idx][1, port_idx])
            time_in_shift_tailored[len_forward_int:-len_backward_int if len_backward_int > 0 else None, :, :] = 0
            
            # Collect PDP for MMSE coefficients calculation
            pdp_tmp = np.zeros((len_forward_int + len_backward_int, time_in_shift_tailored.shape[1], time_in_shift_tailored.shape[2]), dtype=np.complex128)
            if len_backward_int > 0:
                pdp_tmp[:len_backward_int, :, :] = time_in_shift_tailored[-len_backward_int:, :, :]
            pdp_tmp[len_backward_int:len_backward_int + len_forward_int, :, :] = time_in_shift_tailored[:len_forward_int, :, :]
            
            pdp_info[ue_idx][i_port] = np.mean(np.abs(pdp_tmp)**2, axis=(1, 2))
            energy_sig[ue_idx][port_idx] = np.sum(np.mean(np.abs(time_in_shift_tailored)**2, axis=(1, 2)))
    
    # Calculate the timing offset for MSE calculation
    for ue_idx in ue_srs_transmit:
        srs_index_slot = simu_params['UE'][ue_idx]['SRSIndexSlot']
        scs_srs = np.arange(srs_index_slot.shape[0]) - num_idft // 2 - 1
        mean_est_offset = np.mean(list(est_offset[ue_idx]))
        phasor_timing_uncertainty_esti[ue_idx] = np.exp(1j * 2 * np.pi * mean_est_offset / num_idft * scs_srs)
    
    return (delay_in_samples_srs, delay_spread_in_samples_srs, n_leak_cs, 
            phasor_timing_uncertainty_esti, n_point_retain_info, est_offset_all, 
            interval, pdp_info, energy_sig)


# Test function to validate the implementation
def test_srs_timing_delay_estimation():
    """
    Test function for SRS timing and delay spread estimation.
    Creates synthetic test data and validates the function execution.
    """
    print("Testing SRS Timing and Delay Spread Estimation...")
    
    # Create synthetic test parameters
    num_ues = 2
    num_ports = 2
    num_samples = 1024
    num_antennas = 4
    num_slots = 1
    num_idft = 512
    n_cs_max_srs = 8
    
    # Mock simulation parameters
    simu_params = {
        'UE': {},
        'NrUsedforTimingDelayEst': 2
    }
    
    for ue_idx in range(num_ues):
        simu_params['UE'][ue_idx] = {
            'n_csi_SRS': np.random.randint(0, n_cs_max_srs, num_ports),
            'SRSTXSequence': (np.random.randn(64, 1) + 1j * np.random.randn(64, 1)).astype(np.complex128),
            'SRSIndexSlot': np.random.randint(0, 100, (64, 1))
        }
    
    # Create test data
    ue_srs_transmit = list(range(num_ues))
    ue_group_port = {(0, i): list(range(num_ports)) for i in range(num_ues)}
    n_rx = num_antennas
    time_in = (np.random.randn(num_samples, num_antennas, num_slots) + 
               1j * np.random.randn(num_samples, num_antennas, num_slots)).astype(np.complex128)
    is_fft = True
    n_cs_loop = np.arange(n_cs_max_srs)
    ue_idx_loop = np.repeat(ue_srs_transmit, len(n_cs_loop) // len(ue_srs_transmit) + 1)[:len(n_cs_loop)]
    port_idx_loop = np.tile(range(num_ports), len(n_cs_loop) // num_ports + 1)[:len(n_cs_loop)]
    i_group = 0
    ts = 1e-6  # 1 microsecond sampling period
    est_offset_all = []
    
    try:
        # Run the estimation function
        results = estimate_timing_and_delay_spread(
            simu_params, ue_srs_transmit, ue_group_port, n_rx, time_in,
            num_idft, is_fft, n_cs_loop, ue_idx_loop, port_idx_loop,
            i_group, ts, n_cs_max_srs, est_offset_all
        )
        
        # Unpack results
        (delay_in_samples_srs, delay_spread_in_samples_srs, n_leak_cs,
         phasor_timing_uncertainty_esti, n_point_retain_info, est_offset_all_out,
         interval, pdp_info, energy_sig) = results
        
        print("✓ Function executed successfully!")
        print(f"✓ Processed {len(delay_in_samples_srs)} UEs")
        print(f"✓ Total estimated offsets: {len(est_offset_all_out)}")
        
        # Display some results
        for ue_idx in ue_srs_transmit:
            print(f"UE {ue_idx}:")
            print(f"  Timing delays: {delay_in_samples_srs[ue_idx]}")
            print(f"  Delay spreads: {delay_spread_in_samples_srs[ue_idx]}")
            print(f"  Signal energies: {energy_sig[ue_idx]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    # Run test when script is executed directly
    success = test_srs_timing_delay_estimation()
    if success:
        print("\n🎉 SRS Timing and Delay Spread Estimation function is ready for integration!")
    else:
        print("\n❌ Function needs debugging before integration.")
