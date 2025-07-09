# aug_v3_bath.py  
##### batch process augmented data ########################
# input dir, file  
# output dir, file  
# defined in main() function, bottom of file.  
# 
###########################################################

# Imports 
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

import os
import glob
import re
from pathlib import Path

###### Augmenter class definition #########################

class ECGAugmenterV3:
    """
    A class to apply realistic augmentations to 12-lead ECG data based on
    the "ECG Data Augmentation Strategy for Prototype Device" document.

    This class simulates two main types of real-world imperfections:
    1. Lead Displacement: Simulates physical shifts of electrodes.
    2. Noise: Adds various types of noise common in wearable devices.

    Args:
        sampling_rate (int): The sampling rate of the ECG signals (e.g., 125 Hz).
        lead_names (list): A list of 12 lead names in the correct order.
                           Example: ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                                     'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    """
    def __init__(self, sampling_rate=125, lead_names=None):
        self.sampling_rate = sampling_rate
        if lead_names is None:
            self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        else:
            self.lead_names = lead_names

        # Create a mapping from lead name to index for easy lookup
        self.lead_map = {name: i for i, name in enumerate(self.lead_names)}

    def _get_lead(self, ecg, lead_name):
        """Helper to get a lead by name."""
        return ecg[self.lead_map[lead_name], :]

    def _set_lead(self, ecg, lead_name, data):
        """Helper to set a lead by name."""
        ecg[self.lead_map[lead_name], :] = data

    def simulate_lead_displacement(self, ecg):
        """
        Simulates the physical displacement of ECG electrodes by mixing signals
        from adjacent leads. This is Stage 1 of the augmentation pipeline.

        Args:
            ecg (np.ndarray): A 12-lead ECG signal of shape (12, num_samples).

        Returns:
            np.ndarray: The ECG signal with simulated lead displacement.
        """
        displaced_ecg = ecg.copy()

        # --- 1. Limb Leads Augmentation ---
        # Augment leads I and II, then recalculate the rest.
        alpha = np.random.uniform(-0.1, 0.1)
        beta = np.random.uniform(-0.1, 0.1)

        I = self._get_lead(ecg, 'I')
        II = self._get_lead(ecg, 'II')

        # II_aug = (1 - α) * II + α * I
        II_aug = (1 - alpha) * II + alpha * I
        # I_aug = (1 - β) * I + β * II
        I_aug = (1 - beta) * I + beta * II

        self._set_lead(displaced_ecg, 'I', I_aug)
        self._set_lead(displaced_ecg, 'II', II_aug)

        # Recalculate dependent limb leads based on the augmented I and II
        # III = II - I
        self._set_lead(displaced_ecg, 'III', self._get_lead(displaced_ecg, 'II') - self._get_lead(displaced_ecg, 'I'))
        # aVR = -(I + II) / 2
        self._set_lead(displaced_ecg, 'aVR', -(self._get_lead(displaced_ecg, 'I') + self._get_lead(displaced_ecg, 'II')) / 2)
        # aVL = I - II / 2
        self._set_lead(displaced_ecg, 'aVL', self._get_lead(displaced_ecg, 'I') - self._get_lead(displaced_ecg, 'II') / 2)
        # aVF = II - I / 2
        self._set_lead(displaced_ecg, 'aVF', self._get_lead(displaced_ecg, 'II') - self._get_lead(displaced_ecg, 'I') / 2)

        # --- 2. Precordial Leads Augmentation ---
        # V1 is mixed with V2
        # Modify the range from (0, 0.15) to (0, 0.45)
        alpha_v1 = np.random.uniform(0, 0.45)
        v1_aug = (1 - alpha_v1) * self._get_lead(ecg, 'V1') + alpha_v1 * self._get_lead(ecg, 'V2')
        self._set_lead(displaced_ecg, 'V1', v1_aug)

        # Leads V2 to V5 are mixed with their two neighbors
        for i in range(2, 6):
            lead_name = f'V{i}'
            prev_lead = f'V{i-1}'
            next_lead = f'V{i+1}'

            # Modify the range from (0, 0.15) to (0, 0.45)
            alpha = np.random.uniform(0, 0.45)
            # Modify the range from (0, 0.15) to (0, 0.45)
            beta = np.random.uniform(0, 0.45)
            # Ensure coefficients don't sum to > 1
            if alpha + beta > 1:
                # Simple scaling to keep it within bounds while preserving ratio
                total = alpha + beta
                alpha /= total
                beta /= total

            v_orig = self._get_lead(ecg, lead_name)
            v_prev = self._get_lead(ecg, prev_lead)
            v_next = self._get_lead(ecg, next_lead)

            v_aug = (1 - alpha - beta) * v_orig + alpha * v_prev + beta * v_next
            self._set_lead(displaced_ecg, lead_name, v_aug)

        # V6 is mixed with V5
        # Modify the range from (0, 0.15) to (0, 0.45)
        alpha_v6 = np.random.uniform(0, 0.45)
        v6_aug = (1 - alpha_v6) * self._get_lead(ecg, 'V6') + alpha_v6 * self._get_lead(ecg, 'V5')
        self._set_lead(displaced_ecg, 'V6', v6_aug)

        return displaced_ecg

    def simulate_noise(self, ecg):
        """
        Adds various types of realistic noise to the ECG signal.
        This is Stage 2 of the augmentation pipeline.

        Args:
            ecg (np.ndarray): A 12-lead ECG signal of shape (12, num_samples).

        Returns:
            np.ndarray: The ECG signal with added noise.
        """
        noisy_ecg = ecg.copy()
        num_samples = ecg.shape[1]

        # Apply each noise type with a certain probability
        if np.random.rand() < 0.3: # 30% chance to add baseline wander
            noisy_ecg = self.add_baseline_wander(noisy_ecg)

        # Always apply powerline interference for testing
        #if True: # Change probability to 1.0
        #    noisy_ecg = self.add_powerline_interference(noisy_ecg, interval_prob=1.0)

        if np.random.rand() < 0.3: # 30% chance to add powerline
            noisy_ecg = self.add_powerline_interference(noisy_ecg, interval_prob=0.2)

        if np.random.rand() < 0.3: # 30% chance to add muscle artifacts
            noisy_ecg = self.add_muscle_artifacts(noisy_ecg)

        if np.random.rand() < 0.3: # 30% chance to add motion artifacts
            noisy_ecg = self.add_motion_artifacts(noisy_ecg)

        return noisy_ecg

    def add_baseline_wander(self, ecg):
        """Adds low-frequency sine wave noise to simulate breathing."""
        num_samples = ecg.shape[1]
        t = np.arange(num_samples) / self.sampling_rate

        for i in range(ecg.shape[0]): # Apply to each lead
            # Get signal amplitude to scale the noise appropriately
            signal_amplitude = np.max(ecg[i, :]) - np.min(ecg[i, :])
            if signal_amplitude == 0: continue

            # Random frequency for breathing (e.g., 0.05 Hz to 0.5 Hz)
            freq = np.random.uniform(0.05, 0.5)
            # Random amplitude, e.g., 5-15% of signal amplitude
            amplitude = np.random.uniform(0.05, 0.15) * signal_amplitude
            phase = np.random.uniform(0, 2 * np.pi)

            wander = amplitude * np.sin(2 * np.pi * freq * t + phase)
            ecg[i, :] += wander
        return ecg

    def add_powerline_interference(self, ecg, powerline_freq=60.0, interval_duration=0.1, interval_prob=0.2):
        """
        Adds powerline noise (50 or 60 Hz) in intermittent intervals.

        Args:
            ecg (np.ndarray): The ECG signal.
            powerline_freq (float): Frequency of the powerline noise.
            interval_duration (float): Duration of each noise interval in seconds.
            interval_prob (float): Probability of a noise interval occurring at any given point.
        """
        num_samples = ecg.shape[1]
        t = np.arange(num_samples) / self.sampling_rate
        interval_samples = int(interval_duration * self.sampling_rate)

        for i in range(ecg.shape[0]):
            signal_amplitude = np.max(ecg[i, :]) - np.min(ecg[i, :])
            if signal_amplitude == 0: continue

            # Amplitude is typically small, e.g., 1-5% of signal amplitude
            amplitude = np.random.uniform(0.01, 0.05) * signal_amplitude
            phase = np.random.uniform(0, 2 * np.pi)

            # Generate the full powerline noise signal
            full_powerline_noise = amplitude * np.sin(2 * np.pi * powerline_freq * t + phase)

            # Apply noise in intermittent intervals
            for start_sample in range(0, num_samples, interval_samples):
                if np.random.rand() < interval_prob:
                    end_sample = min(start_sample + interval_samples, num_samples)
                    ecg[i, start_sample:end_sample] += full_powerline_noise[start_sample:end_sample]

        return ecg


    def add_muscle_artifacts(self, ecg):
        """Adds high-frequency Gaussian noise to simulate EMG/muscle artifacts."""
        for i in range(ecg.shape[0]):
            signal_amplitude = np.max(ecg[i, :]) - np.min(ecg[i, :])
            if signal_amplitude == 0: continue

            # Noise standard deviation as a fraction of signal amplitude
            sigma = np.random.uniform(0.02, 0.07) * signal_amplitude
            noise = np.random.normal(0, sigma, ecg.shape[1])
            ecg[i, :] += noise
        return ecg

    def add_motion_artifacts(self, ecg):
        """Adds sudden, sharp spikes or baseline shifts."""
        num_samples = ecg.shape[1]

        num_artifacts = np.random.randint(1, 4) # Add 1 to 3 artifacts

        for _ in range(num_artifacts):
            artifact_type = np.random.choice(['spike', 'shift'])

            for i in range(ecg.shape[0]): # Apply to all leads consistently
                signal_amplitude = np.max(ecg[i, :]) - np.min(ecg[i, :])
                if signal_amplitude == 0: continue

                pos = np.random.randint(0, num_samples)

                if artifact_type == 'spike':
                    # Sharp Gaussian spike
                    amplitude = np.random.uniform(0.2, 0.8) * signal_amplitude
                    width = np.random.randint(self.sampling_rate // 50, self.sampling_rate // 10)
                    spike = amplitude * np.exp(-((np.arange(num_samples) - pos)**2) / (2 * (width**2)))
                    if np.random.rand() < 0.5: spike *= -1 # Random polarity
                    ecg[i, :] += spike

                elif artifact_type == 'shift':
                    # Baseline shift
                    magnitude = np.random.uniform(0.1, 0.4) * signal_amplitude
                    if np.random.rand() < 0.5: magnitude *= -1
                    ecg[i, pos:] += magnitude
        return ecg

    def augment(self, ecg):
        """
        Applies the full two-stage augmentation pipeline to an ECG signal.

        Returns:
            np.ndarray: The fully augmented ECG signal.
        """
        ecg_np = ecg.copy()

        # --- STAGE 1: SIMULATE LEAD DISPLACEMENT ---
        displaced_ecg = self.simulate_lead_displacement(ecg_np)

        # --- STAGE 2: SIMULATE NOISE ---
        augmented_ecg = self.simulate_noise(displaced_ecg)

        return augmented_ecg

##########################################################








##########################################################

def main():
    """
    Main function to find .mat files, process them, and save the results.
    Read .mat file, 125 hertz, millivolts (downsampled)
    """
    SAMPLING_RATE = 125  # Hertz, downsampled from 500 PTBXL original  

    # Define the input and output directories
    input_dir = 'large_files/g1_dm/'
    output_dir = 'large_files/g1_aug/'

    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)

    # Find all files with the .mat suffix in the input directory
    search_path = os.path.join(input_dir, '*.mat')
    file_list = glob.glob(search_path)
    
    if not file_list:
        print(f"No *.mat files found in the directory: {input_dir}")
        return

    print(f"Found {len(file_list)} .mat files to process.")

    # Process each file in the list
    for file_path in file_list:
        # Get the base filename without the extension
        base_filename = Path(file_path).stem
        
        # Construct the output filename
        output_filename = f"{base_filename}_aug.mat"
        output_path = os.path.join(output_dir, output_filename)
        
	# Read-in *.mat data from file  
        mat_data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        ecg_data = mat_data['val']

	# Show elements of ecg_data (ndarray type)
	# print('last 6 elements Vs: \n', ecg_data[-6:])
	# print('shape:', ecg_data.shape)

        # Instantiate and execute the processing class
        augmenter = ECGAugmenterV3(sampling_rate=SAMPLING_RATE)
        original_ecg = ecg_data
        augmented_ecg_np = augmenter.augment(original_ecg)

	# save ouptut to *.mat file  
        mat_dict = {'val': augmented_ecg_np}
        scipy.io.savemat(output_path, mat_dict)
        print("-" * 20)

if __name__ == "__main__":
    # This block will only execute when the script is run directly
    main()