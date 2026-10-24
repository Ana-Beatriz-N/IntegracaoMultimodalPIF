"""Testa alinhamento, seleção no treino e cobertura OOF usando dados reais."""
import tempfile
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from proteomica_pipeline import carregar_tabela, SelecionarTreino, ARQUIVO


def verificar():
    x,y,vazias = carregar_tabela(ARQUIVO)
    assert x.index.equals(y.index) and x.columns.is_unique
    assert x.shape == (17,3384) and vazias == 606
    assert y.value_counts().to_dict() == {1:11,0:6}
    a = x.mask(x.eq(0)).to_numpy(float)
    vistos = np.zeros(len(x),int)
    for tr,va in StratifiedKFold(3,shuffle=True,random_state=42).split(x,y):
        s = SelecionarTreino().fit(a[tr],y.iloc[tr])
        antes = s.imputador_.statistics_.copy()
        adulterado = a[va].copy(); adulterado[:] = 1e9
        assert s.transform(adulterado).shape[1] <= 10
        np.testing.assert_array_equal(antes,s.imputador_.statistics_)
        assert np.isfinite(s.transform(a[va])).all()
        vistos[va] += 1
    assert (vistos == 1).all()
    # Duplicações não são agregadas silenciosamente.
    import pandas as pd
    d = pd.read_csv(ARQUIVO,sep='\t',dtype=str,keep_default_na=False)
    d.loc[2,'Protein'] = d.loc[1,'Protein']
    with tempfile.TemporaryDirectory() as pasta:
        p = Path(pasta)/'duplicados.tsv'; d.to_csv(p,sep='\t',index=False)
        try:
            carregar_tabela(p)
        except ValueError:
            pass
        else:
            raise AssertionError('Duplicação não rejeitada')
    print('Validado: rótulos, matriz real, duplicações, seleção no treino e cobertura OOF.')


if __name__ == '__main__':
    verificar()
