import os
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse

# ====== 基础路径配置 ======
subject = 'sample'  # 替换为你自己的 FreeSurfer subject ID
subjects_dir = '/path/to/freesurfer/subjects'  # 替换为实际路径
raw_path = '/path/to/raw_file.fif'  # 替换为你的 EEG 数据路径

# ====== Step 1: 读取原始数据 ======
raw = mne.io.read_raw_fif(raw_path, preload=True)
raw.filter(1., 40.)

# ====== Step 2: 提取事件与计算 ERP（Evoked） ======
events = mne.find_events(raw, stim_channel='STI 014')
event_id = dict(stim=1)  # 修改为实际触发值
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5,
                    baseline=(None, 0), preload=True)
evoked = epochs.average()

# ====== Step 3: 设置头模和电极空间 ======
trans = f'{subjects_dir}/{subject}/mri/trans.fif'  # 共配准文件
src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
model = mne.make_bem_model(subject=subject, ico=4, conductivity=(0.3,), subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

# ====== Step 4: 计算正向解 (forward model) ======
fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem,
                                eeg=True, meg=False, mindist=5.0, n_jobs=1)

# ====== Step 5: 计算逆解算器 (Inverse Operator) ======
noise_cov = mne.compute_covariance(epochs, tmax=0, method='empirical')
inverse_operator = make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2, depth=0.8)

# ====== Step 6: 应用逆解，获取源活动时间序列 ======
snr = 3.0
lambda2 = 1.0 / snr ** 2
stc = apply_inverse(evoked, inverse_operator, lambda2, method='MNE')

# ====== Step 7: 可视化源活动 ======
brain = stc.plot(subject=subject, subjects_dir=subjects_dir, initial_time=0.1,
                 hemi='split', surface='inflated', time_unit='s')
