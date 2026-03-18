import os
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import glob

DATA_DIR = r'./SEED/ExtractedFeatures'
MODEL_DIR = './seed_MVTN_modelparameter'
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CHANNELS = 62
N_BANDS = 5
SEED_LABELS_RAW = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
LABEL_MAP = {1: 0, 0: 1, -1: 2}
SEED_LABELS_MAPPED = [LABEL_MAP[l] for l in SEED_LABELS_RAW]


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GRL, self).__init__()
        self.lambda_ = torch.tensor(lambda_, dtype=torch.float32)

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_)


class FeatureTower(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super(FeatureTower, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class MultiViewDANNMLP(nn.Module):
    def __init__(self, input_size=310, num_classes=3, feature_dim=64):
        super(MultiViewDANNMLP, self).__init__()
        SEED_62_CHANNELS = [
            'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
            'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
            'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
            'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
            'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
            'O2', 'CB2'
        ]
        LEFT_CHANNELS = [
            'FP1', 'AF3', 'F7', 'F5', 'F3', 'F1', 'FT7', 'FC5', 'FC3', 'FC1',
            'T7', 'C5', 'C3', 'C1', 'TP7', 'CP5', 'CP3', 'CP1', 'P7', 'P5',
            'P3', 'P1', 'PO7', 'PO5', 'PO3', 'CB1', 'O1'
        ]
        MIDLINE_CHANNELS = ['FPZ', 'FZ', 'FCZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ']
        RIGHT_CHANNELS = [
            'FP2', 'AF4', 'F8', 'F6', 'F4', 'F2', 'FT8', 'FC6', 'FC4', 'FC2',
            'T8', 'C6', 'C4', 'C2', 'TP8', 'CP6', 'CP4', 'CP2', 'P8', 'P6',
            'P4', 'P2', 'PO8', 'PO6', 'PO4', 'CB2', 'O2'
        ]
        channel_to_index = {ch: i for i, ch in enumerate(SEED_62_CHANNELS)}
        left_ch_indices = sorted([channel_to_index[ch] for ch in LEFT_CHANNELS])
        mid_ch_indices = sorted([channel_to_index[ch] for ch in MIDLINE_CHANNELS])
        right_ch_indices = sorted([channel_to_index[ch] for ch in RIGHT_CHANNELS])
        left_feat_indices = []
        for ch_idx in left_ch_indices:
            left_feat_indices.extend(list(range(ch_idx * N_BANDS, (ch_idx + 1) * N_BANDS)))
        mid_feat_indices = []
        for ch_idx in mid_ch_indices:
            mid_feat_indices.extend(list(range(ch_idx * N_BANDS, (ch_idx + 1) * N_BANDS)))
        right_feat_indices = []
        for ch_idx in right_ch_indices:
            right_feat_indices.extend(list(range(ch_idx * N_BANDS, (ch_idx + 1) * N_BANDS)))
        self.register_buffer('left_idx_tensor', torch.tensor(left_feat_indices, dtype=torch.long))
        self.register_buffer('mid_idx_tensor', torch.tensor(mid_feat_indices, dtype=torch.long))
        self.register_buffer('right_idx_tensor', torch.tensor(right_feat_indices, dtype=torch.long))
        R_INPUT_DIMS = [len(left_feat_indices), len(mid_feat_indices), len(right_feat_indices)]
        B_INPUT_DIM = N_CHANNELS
        self.region_towers = nn.ModuleList([FeatureTower(dim, feature_dim) for dim in R_INPUT_DIMS])
        self.band_towers = nn.ModuleList([FeatureTower(B_INPUT_DIM, feature_dim) for _ in range(N_BANDS)])
        self.fused_feature_dim = feature_dim * (len(self.region_towers) + len(self.band_towers))
        self.emotion_classifier = nn.Linear(self.fused_feature_dim, num_classes)
        self.domain_classifier = nn.Sequential(
            GRL(lambda_=1.0),
            nn.Linear(self.fused_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        r_features = []
        r_data_left = torch.index_select(x, dim=1, index=self.left_idx_tensor)
        r_features.append(self.region_towers[0](r_data_left))
        r_data_mid = torch.index_select(x, dim=1, index=self.mid_idx_tensor)
        r_features.append(self.region_towers[1](r_data_mid))
        r_data_right = torch.index_select(x, dim=1, index=self.right_idx_tensor)
        r_features.append(self.region_towers[2](r_data_right))
        b_features = []
        for i in range(N_BANDS):
            band_indices = torch.arange(i, x.size(1), N_BANDS, device=x.device)
            band_data = torch.index_select(x, dim=1, index=band_indices)
            b_features.append(self.band_towers[i](band_data))
        fused_features = torch.cat(r_features + b_features, dim=1)
        emotion_output = self.emotion_classifier(fused_features)
        domain_output = self.domain_classifier(fused_features)
        return fused_features, emotion_output, domain_output


class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label


def apply_session_zscore_and_normalize_trials(trial_list_raw):
    if not trial_list_raw: return []
    try:
        full_session_data = np.concatenate(trial_list_raw, axis=1)
    except ValueError:
        return []
    mu_s = np.mean(full_session_data, axis=1, keepdims=True)
    std_s = np.std(full_session_data, axis=1, keepdims=True)
    normalized_trials = [(trial_data - mu_s) / (std_s + 1e-6) for trial_data in trial_list_raw]
    return normalized_trials


def extract_1s_samples_for_trial(trial_data_normalized, trial_label):
    trial_data_T = trial_data_normalized.transpose(1, 0, 2)
    num_samples = trial_data_T.shape[0]
    if num_samples == 0:
        return np.array([]), np.array([])
    xs = trial_data_T.reshape(num_samples, -1)
    ys = np.full(num_samples, trial_label)
    return xs, ys


def load_subject_data(subject_file_list):
    data_samples_list = []
    labels_samples_list = []
    if not subject_file_list:
        return None, None

    file_path = subject_file_list[0]
    session_trials_raw = []
    session_trial_labels = []

    try:
        mat_data = sio.loadmat(file_path)
        for k in range(1, 16):
            trial_key = f'de_LDS{k}'
            if trial_key in mat_data:
                trial_data = mat_data[trial_key]
                if trial_data.ndim == 3 and trial_data.shape[0] == 62 and trial_data.shape[2] == 5:
                    session_trials_raw.append(trial_data)
                    session_trial_labels.append(SEED_LABELS_MAPPED[k - 1])
    except Exception:
        return None, None

    if not session_trials_raw:
        return None, None

    normalized_trials = apply_session_zscore_and_normalize_trials(session_trials_raw)
    for trial_data_norm, trial_label in zip(normalized_trials, session_trial_labels):
        X_samples, y_samples = extract_1s_samples_for_trial(trial_data_norm, trial_label)
        if X_samples.size > 0:
            data_samples_list.append(X_samples)
            labels_samples_list.append(y_samples)

    if not data_samples_list:
        return None, None

    X_all = np.concatenate(data_samples_list, axis=0)
    y_all = np.concatenate(labels_samples_list, axis=0)
    return X_all, y_all


def main():
    if not os.path.exists(MODEL_DIR) or not os.path.exists(DATA_DIR):
        return

    mat_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.mat')))
    subject_files = {}
    for mat_file_path in mat_files:
        filename = os.path.basename(mat_file_path)
        if not filename.endswith('.mat') or filename == 'label.mat' or not filename[0].isdigit():
            continue
        subject_id = int(filename.split('_')[0])
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(mat_file_path)

    acc_list = []
    f1_list = []

    for sub_id in range(1, 16):
        model_path = os.path.join(MODEL_DIR, f"subject_{sub_id}.pth")
        if not os.path.exists(model_path):
            continue

        if sub_id not in subject_files:
            continue

        X_test, y_test = load_subject_data(subject_files[sub_id])
        if X_test is None:
            continue

        test_dataset = EEGDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = MultiViewDANNMLP(input_size=310, num_classes=3).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(DEVICE)
                _, emotion_outputs, _ = model(data)
                preds = torch.argmax(emotion_outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        acc_list.append(acc)
        f1_list.append(f1)

    mean_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    mean_f1 = np.mean(f1_list)
    std_f1 = np.std(f1_list)

    print(f"Average Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Average F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")


if __name__ == '__main__':
    main()