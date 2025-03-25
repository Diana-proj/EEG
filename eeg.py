import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne_bids import BIDSPath, read_raw_bids
from mne_icalabel import label_components
from mne.preprocessing import ICA

event_id = {
    'normal': 1,   
    'conflict': 2  
}

'''I have repurposed the below function to output epochs_match and epoch_mismatch variables that we need for time-freq analysis whilst preserving all the preprocessing steps.'''


def process_subject(subject_id, session, bids_root):
    
    # Define file paths
    bids_path = BIDSPath(
        subject=subject_id, session=session, task="PredictionError", suffix="eeg", extension=".vhdr", root=bids_root
    )
    
    # Load raw data
    raw = read_raw_bids(bids_path)

    # Preprocessing steps
    raw.annotations.onset -= 0.063  # Adjust for EEG setup delay
    raw_resampled = raw.copy().resample(sfreq=250, npad="auto")  # Resample
    raw_filtered = raw_resampled.filter(l_freq=1.0, h_freq=124.0).notch_filter(freqs=50)  # Bandpass + Notch filter
    raw_referenced = raw_filtered.set_eeg_reference(ref_channels="average").set_montage("standard_1020")  # Re-reference

    fc1_idx = raw_referenced.ch_names.index('FC1')
    fc2_idx = raw_referenced.ch_names.index('FC2')
    fz_idx = raw_referenced.ch_names.index('Fz')
    cz_idx = raw_referenced.ch_names.index('Cz')

    raw_data = raw_referenced.get_data()

    # Compute the new FCz signal
    fc1_signal = raw_data[fc1_idx, :]
    fc2_signal = raw_data[fc2_idx, :]
    fz_signal = raw_data[fz_idx, :]
    cz_signal = raw_data[cz_idx, :]

    fc_z_signal = (fc1_signal + fc2_signal + fz_signal + cz_signal) / 4

    # Added FCz to the data
    fc_z_info = mne.create_info(['FCz'], raw_referenced.info['sfreq'], ch_types='eeg')
    fc_z_raw = mne.io.RawArray(fc_z_signal[np.newaxis, :], fc_z_info)
    raw_referenced.add_channels([fc_z_raw], force_update_info=True)


    # Update channel names and set position
    raw_referenced.info['chs'][-1]['ch_name'] = 'FCz'
    raw_referenced.rename_channels({raw_referenced.ch_names[-1]: 'FCz'})

    # Set electrode position (estimate using montage)
    raw_referenced.set_montage("standard_1020", match_case=False)

    # Step 5: Extract events based on annotations
    events = []
    for annot in raw_referenced.annotations:
        print(f"Processing annotation: {annot['description']}")
        if 'normal_or_conflict:normal' in annot['description']:
            events.append([int(annot['onset'] * raw_referenced.info['sfreq']), 0, event_id['normal']])
        elif 'normal_or_conflict:conflict' in annot['description']:
            events.append([int(annot['onset'] * raw_referenced.info['sfreq']), 0, event_id['conflict']])
        else:
            print("Skipping irrelevant annotation:", annot['description'])
    events = np.array(events, dtype=int)

    # Extract epochs
    epochs = mne.Epochs(
        raw_referenced, events, event_id=event_id, tmin=-0.3, tmax=0.7,
        baseline=(-0.3, 0), preload=True, event_repeated='merge'
    )
    print(f"Total epochs: {len(epochs)}")

    # Compute Mean Absolute Amplitude for Each Epoch
    epoch_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    mean_amplitudes = np.mean(np.abs(epoch_data), axis=(1, 2))  # Mean over channels and time

    # Rank Epochs by Mean Amplitude
    ranked_indices = np.argsort(mean_amplitudes)  

    # Keep 85% of the Cleanest Epochs
    percentage = 85
    n_epochs_to_keep = int(len(epochs) * (percentage / 100))
    selected_indices = ranked_indices[:n_epochs_to_keep]  # Select top 85% clean epochs

    # Create Clean Epochs and Apply ICA
    clean_epochs = epochs[selected_indices]

    ica = ICA(n_components=10, method='picard', random_state=42, max_iter=5000)
    ica.fit(clean_epochs)

    ica.plot_components()

    from mne_icalabel import label_components

    # Step 6: Use ICLabel for automatic component classification
    labels = label_components(raw_referenced, ica, method='iclabel')

    print("ICLabel Results:")
    for idx, (label, prob) in enumerate(zip(labels['labels'], labels['y_pred_proba'])):
        print(f"Component {idx}: {label} (Probability: {prob:.2f})")

    # Automatically mark bad components for exclusion
    bad_ics = [idx for idx, label in enumerate(labels['labels'])
            if label in ('eye blink', 'muscle artifact', 'line_noise')]

    ica.exclude = bad_ics  # Mark components for exclusion

    ica.apply(raw_referenced)

    raw_referenced.set_meas_date(None)

    #raw_referenced.save(f"cleaned_data_{subject_id}_{session}.fif", overwrite=True)

    # Step 7: Filter ERP data with 0.2 Hz high-pass and 35 Hz low-pass
    raw_referenced = raw_referenced.copy().filter(l_freq=0.2, h_freq=35.0)

    # Step 9: Reject 10% of the noisiest epochs based on signal amplitude
    epoch_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    mean_amplitudes = np.mean(np.abs(epoch_data), axis=(1, 2))  # Compute mean amplitude for each epoch
    threshold = np.percentile(mean_amplitudes, 90)  # Top 10% noisy epochs
    clean_epochs = epochs[mean_amplitudes < threshold]

    # Step 10: Focus analyses on selected electrodes 
    frontal_channels = ['Fz', 'Cz', 'Fp1', 'FC1', 'FC2']
    clean_epochs.pick_channels(frontal_channels)

    # Step 11: Extract ERP negativity peaks (minimum peak) in the 100–300 ms time window
    time_window = (0.1, 0.3)  # 100–300 ms
    negativity_peaks = {}

    for ch_name in frontal_channels:
        channel_idx = clean_epochs.ch_names.index(ch_name)
        erp_data = clean_epochs.average().data[channel_idx]
        times = clean_epochs.times

        # Extract the data within the time window
        mask = (times >= time_window[0]) & (times <= time_window[1])
        time_window_data = erp_data[mask]
        time_window_times = times[mask]

        # Find the minimum (negative) peak
        peak_idx = np.argmin(time_window_data)
        peak_time = time_window_times[peak_idx]
        peak_amplitude = time_window_data[peak_idx]

        negativity_peaks[ch_name] = (peak_time, peak_amplitude)
        print(f"Channel {ch_name}: Negativity peak at {peak_time:.3f} s with amplitude {peak_amplitude:.3f} µV")


    epochs_match = epochs['normal']
    epochs_mismatch = epochs['conflict']
    print(f"Match trials: {len(epochs_match)}, Mismatch trials: {len(epochs_mismatch)}")


    # erp_normal = epochs_match.average()
    # erp_conflict = epochs_mismatch.average()

    # return erp_normal.data, erp_conflict.data
    return epochs_match, epochs_mismatch



'''Below we are plotting the results for only one subject. What needs to be done is the iteration across all subjects.'''

bids_root = "/home/st/st_us-053000/st_st190561/EEG"
subjects = ["02"]
#subjects = ["02", "03", "06", "07", "08", "11", "12", "13", "14", "15", "16"]

sessions = ["Visual", "EMS", "Vibro"]
#sessions = ["Visual"]

channel = 'FCz'

# Initialize lists for storing TFR data
tfr_normal_visual, tfr_conflict_visual = [], []
tfr_normal_ems, tfr_conflict_ems = [], []
tfr_normal_vibro, tfr_conflict_vibro = [], []

for subject in subjects:
    for session in sessions:
        try:
            print(f"Processing Subject {subject}, Session {session}...")

            freqs = np.arange(1, 124, 1)
            n_cycles = freqs / 2

            epochs_match, epochs_mismatch = process_subject(subject, session, bids_root)


            tfr_match = mne.time_frequency.tfr_morlet(epochs_match, freqs=freqs, n_cycles=n_cycles,
                                                      use_fft=True, return_itc=False, decim=3, n_jobs=1)

            tfr_mismatch = mne.time_frequency.tfr_morlet(epochs_mismatch, freqs=freqs, n_cycles=n_cycles,
                                                         use_fft=True, return_itc=False, decim=3, n_jobs=1)


            if session == "Visual":
                tfr_conflict_visual.append(tfr_mismatch.data)
                tfr_normal_visual.append(tfr_match.data)
            elif session == "EMS":
                tfr_conflict_ems.append(tfr_mismatch.data)
                tfr_normal_ems.append(tfr_match.data)
            elif session == "Vibro":
                tfr_conflict_vibro.append(tfr_mismatch.data)
                tfr_normal_vibro.append(tfr_match.data)

        except FileNotFoundError:
            print(f"File not found for subject {subject}, session {session}. Skipping this session.")
            continue


avg_data = {
    "Visual": {
        "normal": np.mean(tfr_normal_visual, axis=0) if tfr_normal_visual else None,
        "conflict": np.mean(tfr_conflict_visual, axis=0) if tfr_conflict_visual else None,
    },
    "EMS": {
        "normal": np.mean(tfr_normal_ems, axis=0) if tfr_normal_ems else None,
        "conflict": np.mean(tfr_conflict_ems, axis=0) if tfr_conflict_ems else None,
    },
    "Vibro": {
        "normal": np.mean(tfr_normal_vibro, axis=0) if tfr_normal_vibro else None,
        "conflict": np.mean(tfr_conflict_vibro, axis=0) if tfr_conflict_vibro else None,
    },
}


tfr_match_avg = tfr_match.copy()
tfr_match_avg.data = avg_data["EMS"]["normal"]  

tfr_mismatch_avg = tfr_mismatch.copy()
tfr_mismatch_avg.data = avg_data["EMS"]["conflict"]


fig, axes = plt.subplots(1, 2, figsize=(14, 8))

print(f"tfr_normal_visual: {len(tfr_normal_visual)}, tfr_conflict_visual: {len(tfr_conflict_visual)}")
print(f"tfr_normal_ems: {len(tfr_normal_ems)}, tfr_conflict_ems: {len(tfr_conflict_ems)}")
print(f"tfr_normal_vibro: {len(tfr_normal_vibro)}, tfr_conflict_vibro: {len(tfr_conflict_vibro)}")

print(f"Total channels in epochs_match: {len(epochs_match.ch_names)}")
print(f"Shape of tfr_match_avg.data: {tfr_match_avg.data.shape}")  # Expected shape: (n_channels, ...)
print(f"Requested channel index: {epochs_match.ch_names.index(channel)}")
print(f"Channel: {channel}")


im1 = axes[0].imshow(tfr_match_avg.data[epochs_match.ch_names.index(channel)], aspect='auto', 
                      origin='lower', extent=[epochs_match.times[0], epochs_match.times[-1], freqs[0], freqs[-1]])
axes[0].set_title("TFR - Match")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(tfr_mismatch_avg.data[epochs_mismatch.ch_names.index(channel)], aspect='auto',
                      origin='lower', extent=[epochs_mismatch.times[0], epochs_mismatch.times[-1], freqs[0], freqs[-1]])
axes[1].set_title("TFR - Mismatch")
fig.colorbar(im2, ax=axes[1])

plt.savefig("/home/st/st_us-053000/st_st190561/EEG/TF1.png")
plt.show()

