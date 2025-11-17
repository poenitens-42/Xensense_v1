import torch
import torch.nn as nn
import numpy as np

class STMLSTMPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=1, horizon=15):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

        self.buffers = {}  # tid → [window of states]

    def update(self, tid, state_vec):
        # state_vec: [x, y, dist, speed, accel]
        if tid not in self.buffers:
            self.buffers[tid] = []
        self.buffers[tid].append(state_vec)
        self.buffers[tid] = self.buffers[tid][-10:]

        if len(self.buffers[tid]) < 10:
            return None  # not enough history yet

        seq = torch.tensor([self.buffers[tid]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out, _ = self.lstm(seq)
            pred = self.fc(out[:, -1, :])
        return float(pred.cpu().item())  # collision probability (0–1)
