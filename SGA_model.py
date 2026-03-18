import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import random
import scipy.io as sio
import glob
import sys

DATA_DIR = r'G:\YWJ\Multimodal_emotions\data\SEED\ExtractedFeatures'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

POPULATION_SIZE = 20
N_GENERATIONS = 30
EPOCHS_GA = 10
EPOCHS_FINAL = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
N_CHANNELS = 62
N_BANDS = 5

SEED_CHANNEL_NAMES = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
    'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

CHANNEL_GROUPS = {
    'Pre-Frontal': ['FP1', 'FPZ', 'FP2'],
    'Left-Frontal': ['AF3', 'F7', 'F5', 'F3', 'F1'],
    'Right-Frontal': ['AF4', 'F8', 'F6', 'F4', 'F2'],
    'Left-Fronto-Temporal': ['FT7', 'FC5', 'FC3', 'FC1'],
    'Right-Fronto-Temporal': ['FT8', 'FC6', 'FC4', 'FC2'],
    'Left-Temporal': ['T7'],
    'Right-Temporal': ['T8'],
    'Left-Central': ['C5', 'C3', 'C1'],
    'Right-Central': ['C6', 'C4', 'C2'],
    'Left-Centro-Parietal': ['TP7', 'CP5', 'CP3', 'CP1'],
    'Right-Centro-Parietal': ['TP8', 'CP6', 'CP4', 'CP2'],
    'Left-Parietal': ['P7', 'P5', 'P3', 'P1'],
    'Right-Parietal': ['P8', 'P6', 'P4', 'P2'],
    'Left-Parieto-Occipital': ['PO7', 'PO5', 'PO3'],
    'Right-Parieto-Occipital': ['PO8', 'PO6', 'PO4'],
    'Occipital': ['O1', 'OZ', 'O2'],
    'Cerebellar': ['CB1', 'CB2'],
    'Midline': ['FZ', 'FCZ', 'CZ', 'CPZ', 'PZ', 'POZ']
}

CHANNEL_TO_IDX = {name: i for i, name in enumerate(SEED_CHANNEL_NAMES)}

CONTRALATERAL_PAIRS = {
    'FP1': 'FP2', 'AF3': 'AF4', 'F7': 'F8', 'F5': 'F6', 'F3': 'F4', 'F1': 'F2',
    'FT7': 'FT8', 'FC5': 'FC6', 'FC3': 'FC4', 'FC1': 'FC2', 'T7': 'T8', 'C5': 'C6',
    'C3': 'C4', 'C1': 'C2', 'TP7': 'TP8', 'CP5': 'CP6', 'CP3': 'CP4', 'CP1': 'CP2',
    'P7': 'P8', 'P5': 'P6', 'P3': 'P4', 'P1': 'P2', 'PO7': 'PO8', 'PO5': 'PO6',
    'PO3': 'PO4', 'O1': 'O2', 'CB1': 'CB2'
}
CONTRALATERAL_PAIRS.update({v: k for k, v in CONTRALATERAL_PAIRS.items()})


class SimpleMLP(nn.Module):
    def __init__(self, input_size=N_CHANNELS * N_BANDS, num_classes=3):
        super(SimpleMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)


class EEGDataset(Dataset):
    def __init__(self, data, labels, channel_mask=None):
        self.raw_data = data
        self.labels = labels
        self.channel_mask = np.array(channel_mask).reshape(N_CHANNELS, 1) if channel_mask is not None else np.ones(
            (N_CHANNELS, 1))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.raw_data[idx] * self.channel_mask
        flat_sample = sample.reshape(-1)
        return torch.tensor(flat_sample, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def apply_session_zscore(trial_list_raw):
    if not trial_list_raw: return []
    try:
        full_session = np.concatenate(trial_list_raw, axis=1)
    except ValueError:
        return []
    mu = np.mean(full_session, axis=1, keepdims=True)
    std = np.std(full_session, axis=1, keepdims=True)
    return [(t - mu) / (std + 1e-6) for t in trial_list_raw]


def load_data(subject_ids):
    data_list, label_list = [], []
    mat_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.mat')))
    subject_map = {}

    for f in mat_files:
        fname = os.path.basename(f)
        if not fname[0].isdigit(): continue
        sid = int(fname.split('_')[0])
        if sid not in subject_map: subject_map[sid] = []
        subject_map[sid].append(f)

    label_map = {1: 0, 0: 1, -1: 2}
    seed_labels = [label_map[l] for l in [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]]

    for sid in subject_ids:
        if sid not in subject_map: continue
        path = subject_map[sid][0]
        try:
            mat = sio.loadmat(path)
            trials = []
            labels = []
            for k in range(1, 16):
                key = f'de_LDS{k}'
                if key in mat:
                    trials.append(mat[key])
                    labels.append(seed_labels[k - 1])

            if not trials: continue
            norm_trials = apply_session_zscore(trials)

            sess_x, sess_y = [], []
            for t_data, t_lbl in zip(norm_trials, labels):
                t_data_T = t_data.transpose(1, 0, 2)
                if t_data_T.shape[0] > 0:
                    sess_x.append(t_data_T)
                    sess_y.append(np.full(t_data_T.shape[0], t_lbl))

            if sess_x:
                data_list.append(np.concatenate(sess_x, axis=0))
                label_list.append(np.concatenate(sess_y, axis=0))
        except Exception:
            continue

    if not data_list: return np.array([]), np.array([])
    return np.concatenate(data_list, axis=0), np.concatenate(label_list, axis=0)


def structured_crossover(p1, p2):
    group = random.choice(list(CHANNEL_GROUPS.keys()))
    c1, c2 = list(p1), list(p2)
    for ch in CHANNEL_GROUPS[group]:
        idx = CHANNEL_TO_IDX[ch]
        c1[idx], c2[idx] = c2[idx], c1[idx]
    return c1, c2


def structured_mutation(c):
    if random.random() < 0.2:
        c = list(c)
        m_type = random.choice(['contralateral', 'group', 'random_flip'])
        if m_type == 'contralateral':
            ch = random.choice(list(CONTRALATERAL_PAIRS.keys()))
            pair = CONTRALATERAL_PAIRS[ch]
            idx1, idx2 = CHANNEL_TO_IDX[ch], CHANNEL_TO_IDX[pair]
            c[idx1] = 1 - c[idx1]
            c[idx2] = 1 - c[idx2]
        elif m_type == 'group':
            group = random.choice(list(CHANNEL_GROUPS.keys()))
            val = 1 if random.random() > 0.5 else 0
            for ch in CHANNEL_GROUPS[group]:
                c[CHANNEL_TO_IDX[ch]] = val
        else:
            idx = random.randrange(N_CHANNELS)
            c[idx] = 1 - c[idx]
        return c
    return c


def fitness_func(chromo, X_train, y_train, X_val, y_val):
    mask = np.array(chromo)
    if mask.sum() == 0: return 0.0

    train_loader = DataLoader(EEGDataset(X_train, y_train, mask), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EEGDataset(X_val, y_val, mask), batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleMLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for _ in range(EPOCHS_GA):
        model.train()
        for d, l in train_loader:
            d, l = d.to(DEVICE), l.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(d), l)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for d, l in val_loader:
            out = model(d.to(DEVICE))
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            labels.extend(l.numpy())

    return accuracy_score(labels, preds)


def run_sga(X_train, y_train, X_val, y_val):
    population = [list(np.random.randint(0, 2, N_CHANNELS)) for _ in range(POPULATION_SIZE)]
    best_fit = -1.0
    best_chromo = population[0]

    for gen in range(N_GENERATIONS):
        scores = [fitness_func(p, X_train, y_train, X_val, y_val) for p in population]
        scores = np.array(scores)

        sorted_idx = np.argsort(scores)[::-1]
        population = [population[i] for i in sorted_idx]
        scores = scores[sorted_idx]

        if scores[0] > best_fit:
            best_fit = scores[0]
            best_chromo = population[0]

        next_pop = []
        top_half = population[:POPULATION_SIZE // 2]
        while len(next_pop) < POPULATION_SIZE:
            p1, p2 = random.sample(top_half, 2)
            c1, c2 = structured_crossover(p1, p2)
            next_pop.extend([structured_mutation(c1), structured_mutation(c2)])
        population = next_pop[:POPULATION_SIZE]

    return best_chromo


def final_eval(chromo, X_train, y_train, X_test, y_test):
    mask = np.array(chromo)
    train_loader = DataLoader(EEGDataset(X_train, y_train, mask), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(EEGDataset(X_test, y_test, mask), batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleMLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for _ in range(EPOCHS_FINAL):
        model.train()
        for d, l in train_loader:
            d, l = d.to(DEVICE), l.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(d), l)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for d, l in test_loader:
            out = model(d.to(DEVICE))
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            labels.extend(l.numpy())

    return accuracy_score(labels, preds), f1_score(labels, preds, average='macro')


if __name__ == '__main__':
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    train_ids = [1, 2, 4, 5, 8, 10, 12, 13, 14, 15]
    val_ids = [6, 9, 11]
    test_ids = [3, 7]

    X_ga_train, y_ga_train = load_data(train_ids)
    X_ga_val, y_ga_val = load_data(val_ids)
    X_test, y_test = load_data(test_ids)

    X_full_train = np.concatenate((X_ga_train, X_ga_val), axis=0)
    y_full_train = np.concatenate((y_ga_train, y_ga_val), axis=0)

    best_mask = run_sga(X_ga_train, y_ga_train, X_ga_val, y_ga_val)

    acc, f1 = final_eval(best_mask, X_full_train, y_full_train, X_test, y_test)

    selected_channels = [SEED_CHANNEL_NAMES[i] for i, val in enumerate(best_mask) if val == 1]

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Optimal Channel Subset: {selected_channels}")