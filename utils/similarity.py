from rdkit import DataStructs
from utils import morgan_fingerprint
from torch.nn import functional as F

def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def cosine_similarity(encoding_a, encoding_b):
    return F.cosine_similarity(encoding_a, encoding_b).item()

def rescaled_cosine_similarity(molecule_a, molecule_b, S, scale="mean"):
    value = cosine_similarity(molecule_a, molecule_b)

    max_ = 1
    min_ = min(S) if scale == "min" else sum(S) / len(S)

    return (value - min_) / (max_ - min_)
