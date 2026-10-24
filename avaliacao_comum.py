"""Métricas binárias compartilhadas pelas modalidades, sem ajuste de modelos."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

METRICAS = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc']


def calcular_metricas(real, previsto, escore):
    """Usa classe positiva 1; especificidade e AUC sem denominador ficam ausentes."""
    if not np.isfinite(escore).all():
        raise ValueError('Escores não finitos: avaliação interrompida.')
    tn, fp, fn, tp = confusion_matrix(real, previsto, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(real, previsto),
                precision=precision_score(real, previsto, zero_division=0),
                recall=recall_score(real, previsto, zero_division=0),
                specificity=tn / (tn + fp) if tn + fp else np.nan,
                f1=f1_score(real, previsto, zero_division=0),
                roc_auc=roc_auc_score(real, escore) if len(np.unique(real)) == 2 else np.nan)
