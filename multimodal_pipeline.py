"""Auditoria genômica e integração conservadora; execute python multimodal_pipeline.py --help.

Não altera o pipeline clínico. A fusão exige previsões independentes e um
manifesto de validação comum; o FASTA sozinho nunca fornece rótulos.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


ALVO = {"0": "Healthy housemate", "1": "Caso incluído"}


def sha256(caminho):
    """Calcula a impressão digital da entrada sem modificá-la."""
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def ler_fasta(caminho):
    """Lê registros FASTA/FA, preservando cabeçalhos e rejeitando texto sem cabeçalho."""
    cabecalho, partes = None, []
    with Path(caminho).open(encoding="utf-8-sig") as arquivo:
        for numero, linha in enumerate(arquivo, 1):
            linha = linha.strip()
            if not linha:
                continue
            if linha.startswith(">"):
                if cabecalho is not None:
                    yield cabecalho, "".join(partes).upper()
                cabecalho, partes = linha[1:].strip(), []
                if not cabecalho:
                    raise ValueError(f"Cabeçalho vazio: {caminho}, linha {numero}")
            else:
                if cabecalho is None:
                    raise ValueError(f"Sequência sem cabeçalho: {caminho}, linha {numero}")
                partes.append("".join(linha.split()))
    if cabecalho is not None:
        yield cabecalho, "".join(partes).upper()


def auditar_genomas(caminhos, k=2):
    """Audita bases e duplicatas; normaliza k-mers somente por janelas ACGT válidas.

    GC e composição usam bases canônicas como denominador. Frações ambíguas e
    cobertura de janelas são exportadas para tornar o denominador explícito.
    Nenhuma sequência é excluída silenciosamente nem interpretada como animal.
    """
    if k not in (1, 2, 3):
        raise ValueError("Use k entre 1 e 3 para limitar a dimensionalidade.")
    auditoria, features = [], []
    vocabulario = ["".join(p) for p in itertools.product("ACGT", repeat=k)]
    for caminho in caminhos:
        for ordem, (cabecalho, seq) in enumerate(ler_fasta(caminho), 1):
            identidade = {"arquivo": Path(caminho).as_posix(), "registro": ordem,
                          "sequence_id": cabecalho.split()[0]}
            contagens = Counter(seq)
            n, canonicas = len(seq), sum(contagens[b] for b in "ACGT")
            ambiguas = sum(contagens[b] for b in "RYSWKMBDHVN")
            invalidas = sum(v for b, v in contagens.items() if b not in "ACGTRYSWKMBDHVN-.")
            auditoria.append({**identidade, "cabecalho": cabecalho, "comprimento": n,
                "bases_canonicas": canonicas, "bases_ambiguas": ambiguas,
                "gaps": contagens["-"] + contagens["."], "caracteres_invalidos": invalidas,
                "sequencia_vazia": n == 0, "sha256_sequencia": hashlib.sha256(seq.encode()).hexdigest()})
            janelas = Counter(seq[i:i+k] for i in range(max(0, n-k+1))
                              if set(seq[i:i+k]) <= set("ACGT"))
            validas = sum(janelas.values())
            features.append({**identidade, "comprimento": n,
                "fracao_ambigua": ambiguas/n if n else np.nan,
                "gc": (contagens["G"]+contagens["C"])/canonicas if canonicas else np.nan,
                **{f"freq_{b}": contagens[b]/canonicas if canonicas else np.nan for b in "ACGT"},
                "janelas_validas": validas,
                "fracao_janelas_validas": validas/(n-k+1) if n >= k else 0,
                **{f"kmer_{m}": janelas[m]/validas if validas else np.nan for m in vocabulario}})
    if not auditoria:
        raise ValueError("Nenhuma sequência encontrada.")
    tabela = pd.DataFrame(auditoria)
    tabela["id_duplicado"] = tabela.sequence_id.duplicated(keep=False)
    tabela["sequencia_duplicada"] = tabela.sha256_sequencia.duplicated(keep=False)
    return tabela, pd.DataFrame(features)


def metricas(y, p):
    """Calcula métricas binárias com limiar pré-fixado em 0,5; AUC exige duas classes."""
    pred = np.asarray(p) >= .5
    return {"n": len(y), "acuracia": accuracy_score(y, pred),
            "precisao": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(set(y)) == 2 else np.nan}


def late_fusion(clinica, genomica, manifesto):
    """Combina probabilidades por ID com pesos fixos iguais e partições comuns.

    Contrato: FIPID, alvo, fold, probabilidade, particao_id nas duas tabelas.
    O manifesto deve atestar OOF, alvo e auditoria conjunta dos grupos de casa,
    animal e similaridade genômica. A declaração não substitui revisão científica.
    Não aceita margens SVM nem previsões do ajuste final na própria coorte.
    """
    if (manifesto.get("alvo") != ALVO or manifesto.get("origem") != "oof"
            or manifesto.get("grupos_conjuntos_auditados") is not True
            or not manifesto.get("documentacao_particoes")):
        raise ValueError("Manifesto requer alvo compatível, OOF e auditoria conjunta documentada.")
    for tabela in (clinica, genomica):
        obrigatorias = {"FIPID", "alvo", "fold", "probabilidade", "particao_id"}
        if not obrigatorias <= set(tabela):
            raise ValueError(f"Colunas obrigatórias: {sorted(obrigatorias)}")
        if tabela[list(obrigatorias)].isna().any().any() or tabela.empty:
            raise ValueError("Previsões vazias ou campos ausentes.")
        if tabela.FIPID.astype(str).str.strip().eq("").any() or tabela.FIPID.duplicated().any():
            raise ValueError("Identificadores vazios/duplicados exigem revisão; não agregar automaticamente.")
        if not tabela.alvo.isin([0, 1]).all():
            raise ValueError("Rótulos devem ser binários.")
        if not np.isfinite(tabela.probabilidade).all() or not tabela.probabilidade.between(0, 1).all():
            raise ValueError("Escores devem ser probabilidades finitas em [0,1].")
        if not tabela.particao_id.eq(manifesto.get("particao_id")).all():
            raise ValueError("Manifesto e probabilidades precisam da mesma partição.")
    pares = clinica.merge(genomica, on="FIPID", how="outer", suffixes=("_clinica", "_genomica"),
                          indicator=True, validate="one_to_one")
    pares["motivo"] = pares["_merge"].astype(str).map({"both": "pareada", "left_only": "sem_genomica", "right_only": "sem_clinica"})
    validas = pares.loc[pares["_merge"].eq("both")].copy()
    for coluna in ("alvo", "fold", "particao_id"):
        if not validas[f"{coluna}_clinica"].eq(validas[f"{coluna}_genomica"]).all():
            raise ValueError(f"Incompatibilidade de {coluna}; fusão interrompida.")
    validas["probabilidade_fusao"] = (validas.probabilidade_clinica + validas.probabilidade_genomica)/2
    linhas = []
    for fold, bloco in validas.groupby("fold_clinica"):
        for modalidade in ("clinica", "genomica", "fusao"):
            linhas.append({"modalidade": modalidade, "fold": fold,
                           **metricas(bloco.alvo_clinica, bloco[f"probabilidade_{modalidade}"])})
    return pares, validas, pd.DataFrame(linhas, columns=["modalidade", "fold", "n", "acuracia", "precisao", "recall", "f1", "roc_auc"])


def executar(args):
    """Executa etapas disponíveis em pasta exclusiva, reutilizando OOF clínico sem reajuste."""
    pasta = Path(args.saida) / datetime.now(timezone.utc).strftime("execucao_%Y%m%dT%H%M%S_%fZ")
    pasta.mkdir(parents=True, exist_ok=False)
    clinica = Path(args.clinica)
    oof = pd.read_csv(clinica / "previsoes_oof.csv", dtype={"FIPID": str})
    referencia = oof.loc[oof.experimento.eq("demografico") & oof.modelo.eq("Regressao_logistica")].copy()
    if referencia.empty or referencia.FIPID.duplicated().any() or not referencia.tipo_escore.eq("probabilidade_nao_calibrada").all():
        raise ValueError("Artefato clínico não atende ao contrato esperado.")
    auditoria, features = auditar_genomas(args.fasta, args.k)
    auditoria.to_csv(pasta / "auditoria_genomica.csv", index=False)
    features.to_csv(pasta / "features_genomicas.csv", index=False)
    referencia.to_csv(pasta / "referencia_clinica_oof.csv", index=False)
    pd.read_csv(clinica / "comparacao_modelos.csv").to_csv(pasta / "metricas_clinicas_historicas.csv", index=False)
    # Não há tabela documental de correspondência: nem igualdade textual de IDs prova vínculo.
    relacao = [{"modalidade": "clinica", "identificador": x, "motivo": "sem_correspondencia_documentada"} for x in referencia.FIPID]
    relacao += [{"modalidade": "genomica", "identificador": x, "motivo": "sem_correspondencia_documentada_e_sem_rotulo"} for x in auditoria.sequence_id]
    pd.DataFrame(relacao).to_csv(pasta / "amostras_nao_pareadas.csv", index=False)
    pd.DataFrame(columns=["FIPID", "sequence_id", "fonte_correspondencia"]).to_csv(pasta / "amostras_pareadas.csv", index=False)
    pd.DataFrame(columns=["modalidade", "fold", "n", "acuracia", "precisao", "recall", "f1", "roc_auc"]).to_csv(pasta / "metricas_multimodais.csv", index=False)
    pd.DataFrame(columns=["FIPID", "probabilidade_fusao"]).to_csv(pasta / "resultados_fusao.csv", index=False)
    contagens = {"registros_genomicos": len(auditoria), "animais_clinicos": len(referencia), "pareados": 0,
        "nao_pareados_clinicos": len(referencia), "nao_pareados_genomicos": len(auditoria),
        "registros_com_id_duplicado": int(auditoria.id_duplicado.sum()),
        "registros_com_sequencia_duplicada": int(auditoria.sequencia_duplicada.sum()),
        "descartados": 0, "status": "sem_validacao_genomica_ou_fusao"}
    (pasta / "contagens.json").write_text(json.dumps(contagens, indent=2, ensure_ascii=False), encoding="utf-8")
    entradas = [*map(Path, args.fasta), clinica / "previsoes_oof.csv", clinica / "configuracao_versoes.json", Path(__file__).resolve()]
    config = {"k": args.k, "semente_futura": 42, "peso_clinico": .5, "limiar": .5,
        "python": platform.python_version(), "alvo": ALVO,
        "versoes": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn")},
        "entradas_sha256": {Path(p).name: sha256(p) for p in entradas}}
    (pasta / "configuracao_versoes.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (pasta / "RELATORIO.md").write_text(
        "# Auditoria multimodal calculada\n\n" + json.dumps(contagens, indent=2, ensure_ascii=False) +
        "\n\nNenhum classificador genômico foi treinado: faltam rótulos e correspondências documentadas. "
        "Nenhuma métrica de fusão foi calculada. Tabelas vazias indicam indisponibilidade, não desempenho zero. "
        "Não houve exclusões nem alterações clínicas. Os resultados clínicos históricos abrangem a coorte original, "
        "não uma comparação multimodal pareada.\n\nO alvo clínico é inclusão no estudo versus Healthy housemate; "
        "não representa virulência ou confirmação de PIF. OOF logístico demográfico foi preservado; "
        "o modelo final ajustado em todos os animais não foi usado para avaliação.\n\n"
        f"Features: k={args.k}, {len(features.columns)-3} descritores incluindo comprimento e indicadores de qualidade. "
        "Não são evidência preditiva. Duplicatas exatas são auditadas; similaridade evolutiva, "
        "alinhamento e agrupamento por linhagem ainda precisam de dados e análise próprios.\n\n"
        "Para avançar: FASTA por amostra independente, tabela sequence_id/FIPID com fonte documental, "
        "animal/casa/coleta, rótulo e critério científico; grupos de similaridade e partições comuns. "
        "Se esses grupos cruzarem os folds clínicos, recalcular somente a avaliação necessária com divisão comum. "
        "Comparar Dummy, regressão logística e Random Forest com pré-processamento dentro dos folds; "
        "limitar dimensão frente ao tamanho da coorte e não ajustar pesos/limiar no teste. "
        "A média 0,5/0,5 é pré-fixada e requer probabilidades comparáveis; avaliar calibração apenas no treino. "
        "Relatar média e dispersão por fold sem interpretar desvio-padrão como intervalo de confiança.\n",
        encoding="utf-8")
    print(pasta.as_posix())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", nargs="+", default=["sequence.fasta"])
    parser.add_argument("--clinica", required=True, help="Pasta existente com previsoes_oof.csv")
    parser.add_argument("--saida", default="resultados_multimodais")
    parser.add_argument("--k", type=int, choices=[1, 2, 3], default=2)
    executar(parser.parse_args())
