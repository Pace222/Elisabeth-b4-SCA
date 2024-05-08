import pickle as pic
from typing import Optional, Callable

import numpy as np
from sklearn.decomposition import PCA

from cpa_utils import HW, HD, corr_coef_vectorized


SIGNAL_STRENGTHS_METHODS = {
    "PCA": lambda path: SignalStrength("PCA", pca_comp, pca_feat, path),
    "SOST": lambda path: SignalStrength("SOST", sost_comp, sig_feat, path),
    "SNR": lambda path: SignalStrength("SNR", snr_comp, sig_feat, path),
    "CORR": lambda path: SignalStrength("CORR", corr_comp, sig_feat, path),
    "DOM": lambda path: SignalStrength("DOM", dom_comp, sig_feat, path),
}

class SignalStrength:
    def __init__(self, name: str, compute_signal_strength: Callable, select_features: Callable, path: Optional[str] = None) -> None:
        self.name = name
        self.compute_signal_strength = compute_signal_strength
        self.select_features = select_features
        self.path = path

        self.num_features = None
        self.sig = None
        if self.path is not None:
            try:
                with open(self.path, "rb") as r:
                    self.sig = pic.load(r)
            except FileNotFoundError:
                pass
        
    def fit(self, traces: np.ndarray, labels: np.ndarray, num_features: int):
        assert traces.shape[0] == labels.shape[0]
        unique_labels = np.unique(labels)
        assert set(unique_labels) == set(range(len(unique_labels)))

        if self.sig is None:
            self.sig = self.compute_signal_strength(traces, labels, num_features)
            if self.path:
                with open(self.path, "wb") as w:
                    pic.dump(self.sig, w)

        self.num_features = num_features
        return self
    
    def transform(self, traces: np.ndarray):
        return self.select_features(self.sig, traces, self.num_features)
    
    def __str__(self) -> str:
        return self.name


def pca_comp(traces: np.ndarray, labels: np.ndarray, num: int):
    if num is None:
        raise ValueError
    return PCA(n_components=num).fit(traces)

def sost_comp(traces: np.ndarray, labels: np.ndarray, num: int):
    unique_labels = np.unique(labels)

    card_g = np.zeros((len(unique_labels), 1), dtype=np.int32)
    m = np.zeros((len(unique_labels), traces.shape[1]), dtype=np.float64)
    v = np.zeros((len(unique_labels), traces.shape[1]), dtype=np.float64)
    for label in unique_labels:
        traces_per_label = traces[labels == label]
        
        card_g[label][0] = len(traces_per_label)
        m[label] = np.mean(traces_per_label, axis=0)
        v[label] = np.var(traces_per_label.astype(np.float64), axis=0, ddof=1)

    f = np.zeros(traces.shape[1], dtype=np.float64)
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            num = m[i] - m[j]
            den = np.sqrt(v[i] / card_g[i] + v[j] / card_g[j])
            f += np.square(num / den)
    
    return f

def snr_comp(traces: np.ndarray, labels: np.ndarray, num: int):
    unique_labels = np.unique(labels)

    m = np.zeros((len(unique_labels), traces.shape[1]), dtype=np.float64)
    v = np.zeros((len(unique_labels), traces.shape[1]), dtype=np.float64)
    for label in unique_labels:
        traces_per_label = traces[labels == label]
        
        m[label] = np.mean(traces_per_label, axis=0)
        v[label] = np.var(traces_per_label.astype(np.float64), axis=0, ddof=1)
    
    return np.mean(np.square(m) - m, axis=0) / np.mean(v, axis=0)

def corr_comp(traces: np.ndarray, labels: np.ndarray, num: int):
    return corr_coef_vectorized(np.array([HW[lab] for lab in labels]), traces)

def dom_comp(traces: np.ndarray, labels: np.ndarray, num: int):
    unique_labels = np.unique(labels)

    m = np.zeros((len(unique_labels), traces.shape[1]), dtype=np.float64)
    for label in unique_labels:
        traces_per_label = traces[labels == label]
        
        m[label] = np.mean(traces_per_label, axis=0)

    f = np.zeros(traces.shape[1], dtype=np.float64)
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            f += m[i] - m[j]
    
    return f


def pca_feat(model: PCA, traces: np.ndarray, num: int):
    return model.transform(traces)

def sig_feat(model: np.ndarray, traces: np.ndarray, num: int):
    features = np.argpartition(model, kth=-num)[-num:]
    return traces[:, features]