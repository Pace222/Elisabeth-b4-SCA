import pickle as pic
from typing import Optional, Callable, Self, Any

import numpy as np
from sklearn.decomposition import PCA

from utils import HW, HD, corr_coef, corr_coef_vectorized

"""
Different signal strength methods
"""
SIGNAL_STRENGTHS_METHODS = {
    "PCA": lambda path: SignalStrength("PCA", pca_comp, pca_feat, path),       # PCA
    "SOST": lambda path: SignalStrength("SOST", sost_comp, sig_feat, path),    # SOST
    "SNR": lambda path: SignalStrength("SNR", snr_comp, sig_feat, path),       # SNR
    "CORR": lambda path: SignalStrength("CORR", corr_comp, sig_feat, path),    # CORR
    "DOM": lambda path: SignalStrength("DOM", dom_comp, sig_feat, path),       # DOM
}

class SignalStrength:
    def __init__(self, name: str, compute_signal_strength: Callable[[np.ndarray, np.ndarray, int], Any], select_features: Callable[[Any, np.ndarray, int], np.ndarray], path: Optional[str] = None) -> None:
        """Class that finds the computes the signal strength of a set of traces.
        Use fit() to compute it.
        Use transform() to return the highest features of the traces.

        Args:
            name (str): Name of the signal strength
            compute_signal_strength (Callable): Function to call to find compute the sigal strength
            select_features (Callable): Function to call to select the highest features of the vector
            path (Optional[str], optional): Path where to store the signal strengths. Defaults to None.
        """
        self.name = name
        self.compute_signal_strength = compute_signal_strength
        self.select_features = select_features
        self.path = path

        self.num_features = None
        self.sig = None
        if self.path is not None:
            # If signal strength has previously been saved, load it (instead of recomputing it in fit())
            try:
                with open(self.path, "rb") as r:
                    self.sig = pic.load(r)
            except FileNotFoundError:
                pass
        
    def fit(self, traces: np.ndarray, labels: np.ndarray, num_features: int) -> Self:
        """Compute the signal strenth of a set of traces

        Args:
            traces (np.ndarray): Traces dataset
            labels (np.ndarray): Labels dataset
            num_features (int): Number of features to select during the feature reduction

        Returns:
            Self: self
        """
        assert traces.shape[0] == labels.shape[0]
        self.num_features = num_features

        # Recompute the signal strength only if necessary
        if self.sig is None:
            self.sig = self.compute_signal_strength(traces, labels, num_features)
            if self.path:
                # Save it
                with open(self.path, "wb") as w:
                    pic.dump(self.sig, w)

        return self
    
    def transform(self, traces: np.ndarray) -> np.ndarray:
        """Selects the relevant features from the traces using the signal strength already computed in the past

        Args:
            traces (np.ndarray): Traces dataset

        Returns:
            np.ndarray: The reduced traces according to the signal strength
        """
        return self.select_features(self.sig, traces, self.num_features)
    
    def __str__(self) -> str:
        """String representation of the signal strength

        Returns:
            str: String representation
        """
        return self.name

def pca_comp(traces: np.ndarray, labels: np.ndarray, num: int) -> PCA:
    """Computes the signal strength with PCA

    Args:
        traces (np.ndarray): Traces dataset
        labels (np.ndarray): Labels dataset
        num (int): Number of features to select during the feature reduction

    Returns:
        PCA: A fitted PCA on the traces with `num` components
    """
    return PCA(n_components=num).fit(traces)

def sost_comp(traces: np.ndarray, labels: np.ndarray, num: int) -> np.ndarray:
    """Computes the SOST signal strength

    Args:
        traces (np.ndarray): Traces dataset
        labels (np.ndarray): Labels dataset
        num (int): Number of features to select during the feature reduction

    Returns:
        np.ndarray: The SOST signal strength
    """
    unique_labels = np.unique(labels)

    card_g = np.zeros((len(unique_labels), 1), dtype=np.int32)             # Number of traces for each label
    m = np.zeros((len(unique_labels), traces.shape[1]), dtype=np.float64)  # Mean of traces for each label
    v = np.zeros((len(unique_labels), traces.shape[1]), dtype=np.float64)  # Variance of traces for each label
    for i in range(len(unique_labels)):
        traces_per_label = traces[labels == unique_labels[i]]
        
        card_g[i][0] = len(traces_per_label)
        m[i] = np.mean(traces_per_label, axis=0)
        v[i] = np.var(traces_per_label.astype(np.float64), axis=0, ddof=1)

    f = np.zeros(traces.shape[1], dtype=np.float64)
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            numer = m[i] - m[j]
            denom = np.sqrt(v[i] / card_g[i] + v[j] / card_g[j])
            f += np.square(numer / denom)
    
    return f

def snr_comp(traces: np.ndarray, labels: np.ndarray, num: int) -> np.ndarray:
    """Computes the SNR signal strength

    Args:
        traces (np.ndarray): Traces dataset
        labels (np.ndarray): Labels dataset
        num (int): Number of features to select during the feature reduction

    Returns:
        np.ndarray: The SNR signal strength
    """
    unique_labels = np.unique(labels)

    m = np.zeros((len(unique_labels), traces.shape[1]), dtype=np.float64)
    v = np.zeros((len(unique_labels), traces.shape[1]), dtype=np.float64)
    for i in range(len(unique_labels)):
        traces_per_label = traces[labels == unique_labels[i]]
        
        m[i] = np.mean(traces_per_label, axis=0)
        v[i] = np.var(traces_per_label.astype(np.float64), axis=0, ddof=1)
    
    return np.mean(np.square(m) - m, axis=0) / np.mean(v, axis=0)

def corr_comp(traces: np.ndarray, labels: np.ndarray, num: int) -> np.ndarray:
    """Computes the CORR signal strength according to the Hamming Weight of the labels

    Args:
        traces (np.ndarray): Traces dataset
        labels (np.ndarray): Labels dataset
        num (int): Number of features to select during the feature reduction

    Returns:
        np.ndarray: The CORR signal strength according to the Hamming Weight of the labels
    """
    return corr_coef(np.array([HW[lab] for lab in labels]), traces)

def dom_comp(traces: np.ndarray, labels: np.ndarray, num: int) -> np.ndarray:
    """Computes the DOM signal strength

    Args:
        traces (np.ndarray): Traces dataset
        labels (np.ndarray): Labels dataset
        num (int): Number of features to select during the feature reduction

    Returns:
        np.ndarray: The DOM signal strength
    """
    unique_labels = np.unique(labels)

    m = np.zeros((len(unique_labels), traces.shape[1]), dtype=np.float64)
    for i in range(len(unique_labels)):
        traces_per_label = traces[labels == unique_labels[i]]
        
        m[i] = np.mean(traces_per_label, axis=0)

    f = np.zeros(traces.shape[1], dtype=np.float64)
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            f += m[i] - m[j]
    
    return f

def pca_feat(model: PCA, traces: np.ndarray, num: int) -> np.ndarray:
    """Selects the the `num` features from the traces that contain the most information on the labels according to PCA

    Args:
        model (PCA): The previously-fitted PCA
        traces (np.ndarray): Traces dataset
        num (int): Number of features to select during the feature reduction

    Returns:
        np.ndarray: The reduced traces
    """
    return model.transform(traces)

def sig_feat(model: np.ndarray, traces: np.ndarray, num: int) -> np.ndarray:
    """Selects the the `num` features from the traces that contain the most information on the labels according to the signal strength

    Args:
        model (np.ndarray): The previously-computed signal strength
        traces (np.ndarray): Traces dataset
        num (int): Number of features to select during the feature reduction

    Returns:
        np.ndarray: The reduced traces
    """
    features = np.argpartition(model, kth=-num)[-num:] # Select the `num` highest features of the signal strength
    return traces[:, features]