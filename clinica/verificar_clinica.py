"""Verificações de integridade e comportamento de falha do componente clínico."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from clinica.clinica_pipeline import (Configuracao, carregar_dados, criar_coorte,
    construir_atributos, definir_folds, ComponenteClinico, GRUPO, ID, CASA, NASCIMENTO)


def deve_falhar(funcao, mensagem):
    """Exige interrupção explicativa em entradas inseguras para a pesquisa."""
    try:
        funcao()
    except ValueError:
        return
    raise AssertionError(mensagem)


def verificar():
    """Valida invariantes científicos com entradas reais e perturbações controladas."""
    config = Configuracao()
    dados = carregar_dados(config)
    coorte = criar_coorte(dados)
    original = dados.copy(deep=True)
    x = construir_atributos(coorte, config)
    pd.testing.assert_frame_equal(dados, original)
    vistos = np.zeros(len(coorte), dtype=int)
    for treino, validacao in definir_folds(coorte, config):
        assert set(coorte.iloc[treino][CASA]).isdisjoint(coorte.iloc[validacao][CASA])
        assert coorte.iloc[treino].alvo.nunique() == coorte.iloc[validacao].alvo.nunique() == 2
        vistos[validacao] += 1
    assert np.all(vistos == 1)
    assert not set([ID, CASA, GRUPO]).intersection(x.columns)
    alterado = dados.copy(); alterado.loc[0, GRUPO] = "categoria nova ambigua"
    deve_falhar(lambda: criar_coorte(alterado), "Categoria nova não foi rejeitada")
    negativo = coorte.copy(); negativo.loc[negativo.index[0], NASCIMENTO] = "12/31/2099"
    assert np.isnan(construir_atributos(negativo, config).iloc[0].idade_anos)
    invalido = coorte.copy(); invalido.loc[invalido.index[0], NASCIMENTO] = "02/30/2020"
    assert np.isnan(construir_atributos(invalido, config).iloc[0].idade_anos)
    componente = ComponenteClinico(config)
    deve_falhar(lambda: componente.ajustar(coorte, coorte.alvo.iloc[::-1]), "Desalinhamento não detectado")
    componente.ajustar(coorte, coorte.alvo)
    novos = coorte.iloc[:3].copy()
    novos["Breed"] = "raca_nao_observada"
    novos = novos.drop(columns=[GRUPO, "Diagnosis Category", "alvo"])
    saida = componente.prever(novos)
    assert saida[ID].equals(novos[ID])
    assert saida.iloc[:, -1].between(0, 1).all()
    # Serialização deve preservar identidade e previsões.
    with tempfile.TemporaryDirectory() as pasta:
        arquivo = Path(pasta) / "componente.joblib"
        joblib.dump(componente, arquivo)
        pd.testing.assert_frame_equal(saida, joblib.load(arquivo).prever(novos))
        repetidos = dados.copy(); repetidos.loc[1, ID] = repetidos.loc[0, ID]
        csv = Path(pasta) / "repetidos.csv"; repetidos.to_csv(csv, index=False)
        deve_falhar(lambda: carregar_dados(replace(config, arquivo=str(csv))), "ID repetido não rejeitado")
    notebook = json.loads(Path(__file__).with_name("Pif_Dados_Clinicos_Reprodutivel.ipynb").read_text(encoding="utf-8"))
    for numero, celula in enumerate(notebook["cells"]):
        if celula["cell_type"] == "code":
            compile("".join(celula["source"]), f"celula_{numero}", "exec")
    print("Integridade validada: grupos, alinhamento, categorias, datas, previsão inédita, serialização e sintaxe do notebook.")


if __name__ == "__main__":
    verificar()
