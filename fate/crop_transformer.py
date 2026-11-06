"""Fate model after Bürger et al. 2026: per-frame CNN encoder + Transformer over time.

Implements the `FatePredictor` interface of `fate/__init__.py`. Smaller than the paper's
ResNet-50 / 8-layer model so that it fits on a 12 GB GPU shared with other jobs.
"""

import numpy as np
import torch
from torch import nn
from torchvision import models

from . import FATES, CellSequence, metrics, truncate


class CropTransformerNet(nn.Module):
    def __init__(self, dim=256, heads=4, layers=2, max_len=64, pretrained=True):
        super().__init__()
        cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder = nn.Sequential(*list(cnn.children())[:-1], nn.Flatten(), nn.Linear(512, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.1, batch_first=True, norm_first=True),
            layers,
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, len(FATES))

    def forward(self, x, pad):  # x (B, T, C, H, W) float, pad (B, T) True = padding
        B, T = x.shape[:2]
        f = self.encoder(x.flatten(0, 1)).view(B, T, -1)
        # learned position embedding, indexed from the first frame of the (left-aligned) sequence
        f = torch.cat([self.cls.expand(B, -1, -1), f], 1)
        f = f + self.pos[:, : T + 1]
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad], 1)
        return self.head(self.norm(self.transformer(f, src_key_padding_mask=mask)[:, 0]))


def collate(seqs: list[CellSequence], max_len: int, train: bool, device):
    T = min(max(len(s.crops) for s in seqs), max_len)
    B = len(seqs)
    x = torch.zeros(B, T, *seqs[0].crops.shape[1:], dtype=torch.uint8)
    pad = torch.ones(B, T, dtype=torch.bool)
    for i, s in enumerate(seqs):
        c = torch.from_numpy(s.crops[-T:])
        x[i, : len(c)] = c  # left-aligned; padded at the end
        pad[i, : len(c)] = False
    if train:  # same flip/rotation for a whole sequence
        k = np.random.randint(4)
        x = torch.rot90(x, k, (-2, -1))
        if np.random.rand() < 0.5:
            x = x.flip(-1)
    x = (x.to(device).float() / 255 - 0.3) / 0.25
    return x, pad.to(device)


class CropTransformer:
    def __init__(self, epochs=12, batch_size=4, lr=1e-4, max_len=48, device="cuda", seed=0,
                 train_trunc=0, **net_kw):
        # train_trunc > 0: during training drop up to that many final frames from 75% of the
        # sequences, so the model learns to predict the fate that many frames ahead; the best
        # epoch is then chosen on the mean balanced accuracy over several truncations.
        self.train_trunc = train_trunc
        self.val_truncs = (0, 5, 10, 20) if train_trunc else (0,)
        self.epochs, self.batch_size, self.lr, self.max_len = epochs, batch_size, lr, max_len
        self.device, self.seed, self.net_kw = device, seed, net_kw

    def fit(self, train, val=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.net = CropTransformerNet(max_len=self.max_len, **self.net_kw).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-3)
        y = np.array([s.fate for s in train])
        w = torch.tensor(len(y) / (len(FATES) * np.bincount(y)), dtype=torch.float32, device=self.device)
        lossf = nn.CrossEntropyLoss(weight=w)
        scaler = torch.amp.GradScaler()
        self.best_, best_state = -1, None
        for ep in range(self.epochs):
            self.net.train()
            perm = np.random.permutation(len(train))
            tot = 0
            for i in range(0, len(perm), self.batch_size):
                b = [train[j] for j in perm[i : i + self.batch_size]]
                if self.train_trunc:
                    b = [truncate([q], np.random.randint(1, self.train_trunc + 1))[0]
                         if np.random.rand() < 0.75 and len(q.crops) > self.train_trunc + 3 else q
                         for q in b]
                x, pad = collate(b, self.max_len, True, self.device)
                t = torch.tensor([s.fate for s in b], device=self.device)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = lossf(self.net(x, pad).float(), t)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * len(b)
            msg = f"epoch {ep + 1}/{self.epochs} train loss {tot / len(train):.3f}"
            if val:
                ms = []
                for t in self.val_truncs:
                    vt = [q for q in truncate(val, t) if len(q.crops) >= 3]
                    ms.append(metrics(np.array([q.fate for q in vt]), self.predict_proba(vt)))
                score = float(np.mean([m["balanced_acc"] for m in ms]))
                msg += " | val bal.acc " + "/".join(f"{m['balanced_acc']:.3f}" for m in ms) \
                    + " auroc " + "/".join(f"{m['auroc']:.3f}" for m in ms) \
                    + f" (truncations {self.val_truncs})"
                if score > self.best_:
                    self.best_ = score
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            print(msg, flush=True)
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "max_len": self.max_len, "net_kw": self.net_kw}, path)

    @classmethod
    def load(cls, path, device="cuda"):
        ck = torch.load(path, map_location=device)
        m = cls(max_len=ck["max_len"], device=device, **ck["net_kw"])
        m.net = CropTransformerNet(max_len=ck["max_len"], pretrained=False, **ck["net_kw"]).to(device)
        m.net.load_state_dict(ck["state"])
        return m

    @torch.no_grad()
    def predict_proba(self, seqs):
        self.net.eval()
        out = []
        order = np.argsort([len(s.crops) for s in seqs])  # similar lengths together
        for i in range(0, len(seqs), 8):
            b = [seqs[j] for j in order[i : i + 8]]
            x, pad = collate(b, self.max_len, False, self.device)
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(torch.softmax(self.net(x, pad).float(), 1).cpu().numpy())
        p = np.concatenate(out)
        res = np.empty_like(p)
        res[order] = p
        return res

