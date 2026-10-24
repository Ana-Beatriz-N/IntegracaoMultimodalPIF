from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
import joblib
from avaliacao_comum import METRICAS, calcular_metricas


ARQUIVO_PADRAO = "06.2025_-_FIP_and_Housemate_Clinical_Data_Final_-_for_Dryad_Upload.csv"
CASA, ID = "Home number ID", "FIPID"
GRUPO, DIAGNOSTICO = "Included/excluded and group", "Diagnosis Category"
TRATAMENTO = "Treated at time of enrollment (Y/N) If so, how long?"
COLETA, NASCIMENTO = "Date samples collected", "DOB"
FORMA = "Predominant clinical form (wet, dry, optho, neuro, combo)"
SORO, PCR = "FCoV Serology/IFA", "FCoV PCR and sample type"
COLUNAS = [CASA, ID, GRUPO, DIAGNOSTICO, TRATAMENTO, COLETA, "Sex", NASCIMENTO,
           "Breed", FORMA, SORO, PCR]
EXCLUSOES = {
    "exclude - on treatment", "exclude - not fip",
    "exclude - previously treated and in remission", "exclude - poor sample quality",
    "exclude - poor sample quality; resampled gandhi for fip011",
    "healthy housemate - exclude bc on treatment",
}
AUSENTES = {"", "na", "n/a", "nan", "unk", "unknown"}
BASE = ["idade_anos", "sexo", "raca"]
EXPERIMENTOS = {
    "demografico": BASE,
    "com_forma_exploratorio": BASE + ["forma_clinica"],
    "com_exames_exploratorio": BASE + ["forma_clinica", "sorologia", "pcr"],
    "com_tratamento_exploratorio": BASE + ["forma_clinica", "sorologia", "pcr", "tratamento"],
}
ROTULOS = ["Healthy housemate", "Caso incluído"]


@dataclass
class Configuracao:
    """Parâmetros definidos antes de observar as métricas dos modelos."""
    arquivo: str = ARQUIVO_PADRAO
    resultados: str = "resultados_clinicos"
    semente: int = 42
    folds: int = 5
    idade_maxima: float = 30.0  # Limite operacional de auditoria, não critério diagnóstico.
    repeticoes_importancia: int = 10
    permutacoes_associacao: int = 1999
    arvores: int = 200
    modelos: tuple = ("Dummy", "Regressao_logistica", "SVM", "Random_forest")


def normalizar(valor):
    """Normaliza espaços e caixa sem inferir equivalências clínicas."""
    return " ".join(str(valor).split()).casefold()


def salvar_tabela(tabela, destino):
    """Exporta CSV UTF-8 com índice explícito para manter rastreabilidade."""
    tabela.to_csv(destino, index=True, encoding="utf-8-sig")


def carregar_dados(config):
    """Lê tudo como texto e preserva NA literal para auditar o significado original."""
    caminho = Path(config.arquivo)
    if not caminho.is_file():
        raise FileNotFoundError(f"CSV não encontrado: {caminho.resolve()}. Informe --arquivo CAMINHO.")
    dados = pd.read_csv(caminho, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if dados.columns.tolist() != COLUNAS:
        raise ValueError(f"Esquema diferente do esperado. Colunas observadas: {dados.columns.tolist()}")
    if dados.empty:
        raise ValueError("O CSV não contém registros.")
    if dados[ID].map(normalizar).isin(AUSENTES).any() or dados[ID].duplicated().any():
        raise ValueError("FIPID ausente ou repetido: investigar animal/recoleta antes de modelar.")
    if dados[CASA].map(normalizar).isin(AUSENTES).any():
        raise ValueError("Casa ausente: não é possível garantir separação dos grupos.")
    return dados


def auditar_dados(dados, pasta):
    """Documenta tipos, duplicações, categorias e equivalências sem modificar a entrada."""
    falta = dados.map(lambda x: normalizar(x) in AUSENTES)
    auditoria = pd.DataFrame({"tipo_original": dados.dtypes.astype(str),
        "ausentes_n": falta.sum(), "ausentes_percentual": falta.mean() * 100,
        "categorias_originais_n": dados.nunique()})
    salvar_tabela(auditoria, pasta / "auditoria_colunas.csv")
    salvar_tabela(dados.head(10), pasta / "amostras_iniciais.csv")
    salvar_tabela(dados.describe(include="all"), pasta / "descritivas_originais.csv")
    salvar_tabela(dados[dados.duplicated(keep=False)], pasta / "duplicados_exatos.csv")
    salvar_tabela(dados.groupby(CASA).agg(animais=(ID, "size")), pasta / "animais_por_casa.csv")
    categorias = []
    for coluna in dados:
        for valor, n in dados[coluna].value_counts(dropna=False).items():
            categorias.append({"coluna": coluna, "original": valor,
                               "normalizado": normalizar(valor), "n": n})
    salvar_tabela(pd.DataFrame(categorias), pasta / "categorias_originais_e_normalizadas.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    auditoria.ausentes_percentual.sort_values().plot.barh(ax=ax)
    ax.set(xlabel="Valores ausentes (%)", title="Ausência semântica no CSV original (n=100 se entrada original)")
    fig.tight_layout(); fig.savefig(pasta / "ausentes.png", dpi=160); plt.close(fig)
    return auditoria


def criar_coorte(dados, pasta=None):
    """Valida categorias e define 1=Include, 0=Healthy housemate, sem confirmar PIF."""
    grupo = dados[GRUPO].map(normalizar)
    desconhecidos = set(grupo) - EXCLUSOES - {"include", "healthy housemate"}
    if desconhecidos:
        raise ValueError(f"Categorias de inclusão desconhecidas/ambíguas: {desconhecidos}. Revisar protocolo.")
    incluir = grupo.isin(["include", "healthy housemate"])
    coorte = dados.loc[incluir].copy()
    diag = coorte[DIAGNOSTICO].map(normalizar)
    esperado = np.where(grupo.loc[incluir].eq("include"),
                        diag.isin(["very likely", "highly suspicious", "suspicious"]),
                        diag.eq("na"))
    if not esperado.all():
        raise ValueError("Combinação grupo/diagnóstico inesperada. Revisar antes de criar o alvo.")
    coorte["alvo"] = grupo.loc[incluir].map({"include": 1, "healthy housemate": 0}).astype(int)
    if coorte.alvo.nunique() != 2:
        raise ValueError("São necessárias duas classes para classificação binária.")
    if pasta is not None:
        fluxo = dados[[ID, CASA, GRUPO, DIAGNOSTICO]].copy()
        fluxo["incluido_na_analise"] = incluir
        salvar_tabela(fluxo, pasta / "fluxo_inclusao.csv")
        salvar_tabela(pd.crosstab(coorte[GRUPO], coorte[DIAGNOSTICO]), pasta / "grupo_diagnostico.csv")
        salvar_tabela(coorte.alvo.value_counts().rename("n").to_frame(), pasta / "distribuicao_alvo.csv")
    return coorte


def construir_atributos(dados, config, pasta=None):
    """Aplica regras fixas por registro, sem aprender parâmetros antes da validação.

    Datas usam mês/dia/ano, coerente com valores como 11/23/2021 do CSV.
    NA não é convertido em resultado negativo. Exames não realizados são mantidos
    como categoria explícita porque refletem o processo de investigação.
    """
    x = pd.DataFrame(index=dados.index)
    problemas = []
    datas = {}
    for coluna in [COLETA, NASCIMENTO]:
        serie = dados[coluna].map(normalizar)
        datas[coluna] = pd.to_datetime(serie.mask(serie.isin(AUSENTES)), format="%m/%d/%Y", errors="coerce")
        for indice in dados.index[~serie.isin(AUSENTES) & datas[coluna].isna()]:
            problemas.append({"FIPID": dados.loc[indice, ID], "campo": coluna,
                              "valor": dados.loc[indice, coluna], "motivo": "Data inválida"})
    idade = (datas[COLETA] - datas[NASCIMENTO]).dt.days / 365.25
    invalida = (idade < 0) | (idade > config.idade_maxima)
    futura = (datas[COLETA] > pd.Timestamp.now().normalize()) | (datas[NASCIMENTO] > pd.Timestamp.now().normalize())
    for indice in dados.index[invalida | futura]:
        problemas.append({"FIPID": dados.loc[indice, ID], "campo": "idade_anos",
                          "valor": idade.loc[indice], "motivo": "Idade negativa/implausível ou data futura"})
    x["idade_anos"] = idade.mask(invalida | futura)
    for origem, destino in [("Sex", "sexo"), ("Breed", "raca")]:
        valores = dados[origem].map(normalizar)
        x[destino] = valores.mask(valores.isin(AUSENTES), np.nan)
    if not set(x.sexo.dropna()).issubset({"mn", "fs", "mi", "fi"}):
        raise ValueError("Código de sexo desconhecido: revisar dicionário.")
    # Mantém todas as combinações descritas; não equipara formas suspeitas a diagnóstico.
    forma = dados[FORMA].map(normalizar)
    x["forma_clinica"] = forma.mask(forma.isin(AUSENTES), np.nan)
    for origem, destino in [(SORO, "sorologia"), (PCR, "pcr")]:
        def resultado_exame(valor):
            """Extrai somente resultado qualitativo, sem datas, títulos ou material."""
            texto = normalizar(valor)
            if texto in AUSENTES:
                return np.nan
            if texto in {"not performed", "not perfomed"}:
                return "nao_realizado"
            if texto.startswith("positive"):
                return "positivo"
            if texto.startswith("negative"):
                return "negativo"
            raise ValueError(f"Resultado de exame desconhecido em {origem}: {valor}")
        x[destino] = dados[origem].map(resultado_exame)
    def resultado_tratamento(valor):
        """Preserva tratamento desconhecido; não interpreta NA como ausência de tratamento."""
        texto = normalizar(valor)
        if texto in AUSENTES:
            return np.nan
        if texto in {"no", "n"}:
            return "nao"
        if texto.startswith("yes") or texto == "o started empiric gs day before sample collection with no diagnostics":
            return "sim"
        raise ValueError(f"Tratamento não reconhecido: {valor}")
    x["tratamento"] = dados[TRATAMENTO].map(resultado_tratamento)
    if pasta is not None:
        salvar_tabela(pd.DataFrame(problemas, columns=["FIPID", "campo", "valor", "motivo"]), pasta / "problemas_datas.csv")
        salvar_tabela(pd.concat([dados[[ID, CASA]], x], axis=1), pasta / "atributos_derivados.csv")
        salvar_tabela(x.describe(include="all"), pasta / "descritivas_atributos.csv")
        salvar_tabela(pd.DataFrame({"n": x.isna().sum(), "percentual": x.isna().mean()*100}), pasta / "ausentes_atributos.csv")
    return x


def analisar_coorte(coorte, x, pasta):
    """Exporta distribuições observadas por grupo; ausências ficam visíveis nas figuras."""
    y = coorte.alvo
    for coluna in ["sexo", "raca", "forma_clinica", "sorologia", "pcr", "tratamento"]:
        tabela = pd.crosstab(x[coluna].fillna("ausente/não aplicável"), y).reindex(columns=[0, 1], fill_value=0)
        tabela.columns = ROTULOS
        salvar_tabela(tabela, pasta / f"distribuicao_{coluna}.csv")
        salvar_tabela(tabela.div(tabela.sum(axis=0), axis=1)*100, pasta / f"percentual_{coluna}.csv")
        ax = tabela.plot.barh(figsize=(10, max(4, len(tabela)*0.32)))
        ax.set(xlabel="Número de animais", ylabel=coluna.replace("_", " "), title=f"Distribuição de {coluna.replace('_', ' ')} por grupo")
        ax.figure.tight_layout(); ax.figure.savefig(pasta / f"distribuicao_{coluna}.png", dpi=160); plt.close(ax.figure)
    fig, ax = plt.subplots()
    ax.boxplot([x.loc[y.eq(i), "idade_anos"].dropna() for i in [0, 1]], tick_labels=ROTULOS)
    ax.set(ylabel="Idade na coleta (anos)", title="Idades válidas por grupo; ausências excluídas apenas deste gráfico")
    fig.tight_layout(); fig.savefig(pasta / "idade_grupo.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots()
    y.value_counts().reindex([0, 1]).plot.bar(ax=ax)
    ax.set_xticklabels(ROTULOS, rotation=0)
    ax.set(ylabel="Número de animais", title="Distribuição do alvo conservador")
    fig.tight_layout(); fig.savefig(pasta / "classes.png", dpi=160); plt.close(fig)
    salvar_tabela(x.assign(grupo=y).groupby("grupo").idade_anos.describe(), pasta / "idade_por_grupo.csv")


def associacoes_por_casa(coorte, x, config, pasta):
    """Testa associação por permutação restrita dentro da casa, preservando dependência.

    Casas sem contraste de rótulos não contribuem à permutação. O teste é
    condicional à casa e pressupõe permutabilidade intradomiciliar sob H0.
    Não equivale a inferência causal; faltantes podem refletir seleção.
    """
    rng = np.random.default_rng(config.semente)
    y = coorte.alvo.to_numpy()
    grupos = coorte[CASA].to_numpy()
    indices = [np.flatnonzero(grupos == g) for g in np.unique(grupos)]
    informativas = sum(len(np.unique(y[idx])) == 2 for idx in indices)
    resultados = []
    for coluna in x:
        if coluna == "idade_anos":
            valores = x[coluna].to_numpy()
            validos = np.isfinite(valores)
            def estatistica(rotulos):
                """Diferença média: incluídos menos controles, somente idades válidas."""
                return np.mean(valores[validos & (rotulos == 1)]) - np.mean(valores[validos & (rotulos == 0)])
            if any(np.sum(validos & (y == c)) < 2 for c in [0, 1]):
                continue
            efeito = estatistica(y)
            nome_efeito = "diferenca_media_anos"
            n = int(validos.sum())
        else:
            # Incluir ausência explicita associação do padrão de coleta com o grupo.
            codigos, niveis = pd.factorize(x[coluna].fillna("ausente/não aplicável"))
            if len(niveis) < 2:
                continue
            def estatistica(rotulos):
                """Qui-quadrado usado só como estatística; p vem da permutação agrupada."""
                tabela = np.zeros((len(niveis), 2), dtype=int)
                np.add.at(tabela, (codigos, rotulos), 1)
                return chi2_contingency(tabela, correction=False)[0]
            n = len(y)
            efeito = np.sqrt(estatistica(y) / n)
            nome_efeito = "V_cramer_descritivo"
        observado = abs(estatistica(y))
        extremos = 0
        for _ in range(config.permutacoes_associacao):
            permutado = y.copy()
            for idx in indices:
                permutado[idx] = rng.permutation(y[idx])
            extremos += abs(estatistica(permutado)) >= observado - 1e-12
        resultados.append({"variavel": coluna, "n": n, "efeito": efeito,
            "medida_efeito": nome_efeito, "estatistica_absoluta": observado,
            "casas_com_duas_classes": informativas,
            "p_permutacao": (extremos+1)/(config.permutacoes_associacao+1)})
    tabela = pd.DataFrame(resultados)
    if not tabela.empty:
        p = tabela.p_permutacao.to_numpy()
        ordem = np.argsort(p); m = len(p)
        ajustado = np.minimum.accumulate((p[ordem]*m/np.arange(1, m+1))[::-1])[::-1]
        q = np.empty(m); q[ordem] = np.minimum(ajustado, 1)
        tabela["q_benjamini_hochberg"] = q
    salvar_tabela(tabela, pasta / "associacoes_exploratorias.csv")
    return tabela


def criar_modelos(config):
    """Modelos simples com hiperparâmetros fixos, sem busca ou seleção pelos resultados."""
    modelos = {
        "Dummy": DummyClassifier(strategy="prior"),
        "Regressao_logistica": LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=config.semente),
        # Escore de margem para AUC: evita calibração interna não agrupada do SVC.
        "SVM": SVC(C=1.0, kernel="linear", class_weight="balanced", random_state=config.semente),
        "Random_forest": RandomForestClassifier(n_estimators=config.arvores, max_depth=3,
            min_samples_leaf=4, class_weight="balanced", random_state=config.semente, n_jobs=1),
    }
    return {nome: modelos[nome] for nome in config.modelos}


def criar_pipeline(colunas, modelo):
    """Imputação, codificação e escala são ajustadas exclusivamente ao treino de cada fold."""
    numericas = [c for c in colunas if c == "idade_anos"]
    categoricas = [c for c in colunas if c not in numericas]
    numerico = Pipeline([("imputacao", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("escala", StandardScaler())])
    categorico = Pipeline([("imputacao", SimpleImputer(strategy="constant", fill_value="ausente", keep_empty_features=True)),
                           ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return Pipeline([("preprocessamento", ColumnTransformer([
        ("numerico", numerico, numericas), ("categorico", categorico, categoricas)])),
        ("modelo", clone(modelo))])


def definir_folds(coorte, config):
    """Preserva casas inteiras e exige duas classes em todo treino e validação."""
    y, casas = coorte.alvo, coorte[CASA]
    limite = min(config.folds, int(y.value_counts().min()), casas.nunique(),
                 min(casas[y.eq(c)].nunique() for c in [0, 1]))
    for n in range(limite, 1, -1):
        # Sem shuffle: alocação determinística e sem depender da ordem aleatória das casas.
        cv = StratifiedGroupKFold(n_splits=n, shuffle=False)
        folds = list(cv.split(coorte, y, groups=casas))
        if all(y.iloc[a].nunique() == 2 and y.iloc[b].nunique() == 2 for a, b in folds):
            for treino, teste in folds:
                assert set(casas.iloc[treino]).isdisjoint(casas.iloc[teste])
            return folds
    raise ValueError("Não foi possível formar ao menos dois folds com duas classes e casas separadas.")


def avaliar_modelos(coorte, x, config, pasta):
    """Gera métricas por fold e OOF, sem avaliar ajuste em suas próprias amostras."""
    assert coorte.index.equals(x.index) and coorte[ID].is_unique
    y = coorte.alvo
    folds = definir_folds(coorte, config)
    metricas, previsoes, importancias, parametros, coeficientes = [], [], [], [], []
    divisao = coorte[[ID, CASA, "alvo"]].copy(); divisao["fold"] = -1
    for fold, (_, teste) in enumerate(folds, 1):
        divisao.iloc[teste, divisao.columns.get_loc("fold")] = fold
    salvar_tabela(divisao, pasta / "divisao_folds.csv")
    for experimento, colunas in EXPERIMENTOS.items():
        for nome, modelo in criar_modelos(config).items():
            print(f"Avaliando {experimento} / {nome}", flush=True)
            oof = []
            for fold, (treino, teste) in enumerate(folds, 1):
                pipeline = criar_pipeline(colunas, modelo)
                xt, xv = x.iloc[treino][colunas], x.iloc[teste][colunas]
                yt, yv = y.iloc[treino], y.iloc[teste]
                pipeline.fit(xt, yt)
                pred = pipeline.predict(xv)
                probabilistico = hasattr(pipeline, "predict_proba")
                escore = pipeline.predict_proba(xv)[:, 1] if probabilistico else pipeline.decision_function(xv)
                assert np.isfinite(escore).all()
                metricas.append({"experimento": experimento, "modelo": nome, "fold": fold,
                    "n_treino": len(treino), "n_validacao": len(teste),
                    **calcular_metricas(yv, pred, escore)})
                linhas = coorte.iloc[teste][[ID, CASA, GRUPO, DIAGNOSTICO, "alvo"]].copy()
                linhas = linhas.assign(experimento=experimento, modelo=nome, fold=fold,
                    previsao=pred, escore=escore, tipo_escore="probabilidade_nao_calibrada" if probabilistico else "margem_svm")
                linhas["erro"] = linhas.previsao.ne(linhas.alvo)
                oof.append(linhas)
                nomes = pipeline.named_steps["preprocessamento"].get_feature_names_out()
                parametros.append({"experimento": experimento, "modelo": nome, "fold": fold,
                    "atributos_codificados": nomes.tolist(), "pipeline": str(pipeline),
                    "hiperparametros_modelo": modelo.get_params(),
                    "imputacao_idade": pipeline.named_steps["preprocessamento"].named_transformers_["numerico"].named_steps["imputacao"].statistics_.tolist()})
                if nome != "Dummy":
                    importancia = permutation_importance(pipeline, xv, yv, scoring="roc_auc",
                        n_repeats=config.repeticoes_importancia, random_state=config.semente+fold, n_jobs=1)
                    for j, coluna in enumerate(colunas):
                        for repeticao, valor in enumerate(importancia.importances[j]):
                            importancias.append({"experimento": experimento, "modelo": nome, "fold": fold,
                                "variavel": coluna, "repeticao": repeticao, "queda_auc": valor})
                estimador = pipeline.named_steps["modelo"]
                if hasattr(estimador, "coef_"):
                    for atributo, valor in zip(nomes, estimador.coef_[0]):
                        coeficientes.append({"experimento": experimento, "modelo": nome,
                            "fold": fold, "categoria_codificada": atributo, "coeficiente": float(valor)})
            oof = pd.concat(oof).sort_index()
            assert oof.index.equals(coorte.index) and oof[ID].equals(coorte[ID])
            assert oof.alvo.equals(y)
            previsoes.append(oof)
            matriz = confusion_matrix(oof.alvo, oof.previsao, labels=[0, 1])
            prefixo = f"{experimento}_{nome}"
            salvar_tabela(pd.DataFrame(matriz, index=ROTULOS, columns=ROTULOS), pasta / f"matriz_{prefixo}.csv")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(matriz, display_labels=ROTULOS).plot(ax=ax, colorbar=False)
            ax.set(title=f"Validação OOF: {experimento}\n{nome}", xlabel="Grupo previsto", ylabel="Grupo observado")
            fig.tight_layout(); fig.savefig(pasta / f"matriz_{prefixo}.png", dpi=150); plt.close(fig)
            fig, ax = plt.subplots()
            # Curvas individuais: margens de modelos SVM de folds diferentes não são calibradas entre si.
            for fold, linhas in oof.groupby("fold"):
                RocCurveDisplay.from_predictions(linhas.alvo, linhas.escore, name=f"Fold {fold}", ax=ax)
            ax.set(title=f"ROC na validação de cada fold: {nome}\n{experimento}", xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos")
            fig.tight_layout(); fig.savefig(pasta / f"roc_{prefixo}.png", dpi=150); plt.close(fig)
    metricas = pd.DataFrame(metricas)
    medidas = METRICAS
    resumo = metricas.groupby(["experimento", "modelo"])[medidas].agg(["mean", "std"])
    resumo.columns = [f"{a}_{'media' if b == 'mean' else 'dp'}" for a, b in resumo.columns]
    salvar_tabela(metricas, pasta / "metricas_folds.csv")
    salvar_tabela(resumo, pasta / "comparacao_modelos.csv")
    previsoes = pd.concat(previsoes)
    salvar_tabela(previsoes, pasta / "previsoes_oof.csv")
    salvar_tabela(previsoes[previsoes.erro], pasta / "erros_oof.csv")
    erros = previsoes.groupby([ID, "alvo"]).erro.agg(["sum", "count", "mean"])
    salvar_tabela(erros, pasta / "frequencia_erros_amostra.csv")
    salvar_tabela(pd.DataFrame(coeficientes), pasta / "coeficientes_categorias_por_fold.csv")
    importancia = pd.DataFrame(importancias)
    salvar_tabela(importancia, pasta / "importancia_validacao_repeticoes.csv")
    agregada = importancia.groupby(["experimento", "modelo", "fold", "variavel"]).queda_auc.mean()
    agregada = agregada.groupby(["experimento", "modelo", "variavel"]).agg(["mean", "std"])
    salvar_tabela(agregada, pasta / "importancia_validacao_resumo.csv")
    (pasta / "preprocessamento_folds.json").write_text(json.dumps(parametros, indent=2, ensure_ascii=False), encoding="utf-8")
    for experimento in EXPERIMENTOS:
        tabela = resumo.loc[experimento]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(tabela.index, tabela.roc_auc_media, yerr=tabela.roc_auc_dp, capsize=4)
        ax.axhline(0.5, color="gray", linestyle="--")
        ax.set(ylim=(0, 1.05), ylabel="ROC-AUC: média e desvio entre folds", title=f"Comparação dos modelos: {experimento}")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout(); fig.savefig(pasta / f"comparacao_{experimento}.png", dpi=150); plt.close(fig)
    return resumo, previsoes, metricas


class ComponenteClinico:
    """Interface reutilizável: dados brutos -> ID separado + previsão/escore clínico.

    Ajuste no conjunto completo destina-se a uso posterior, nunca à avaliação
    independente. Não calibra probabilidades nem implementa outra modalidade.
    """
    def __init__(self, config=None, experimento="demografico", modelo="Regressao_logistica"):
        self.config = config or Configuracao()
        self.experimento = experimento
        self.modelo = modelo

    def ajustar(self, dados, alvo):
        """Ajusta dados clínicos com alvo indexado e identificadores únicos."""
        if not dados.index.equals(alvo.index) or not dados[ID].is_unique:
            raise ValueError("Índices desalinhados ou FIPID repetido.")
        if set(alvo.unique()) != {0, 1}:
            raise ValueError("O alvo deve conter as classes 0 e 1.")
        self.colunas_ = EXPERIMENTOS[self.experimento]
        self.pipeline_ = criar_pipeline(self.colunas_, criar_modelos(self.config)[self.modelo])
        self.pipeline_.fit(construir_atributos(dados, self.config)[self.colunas_], alvo)
        return self

    def prever(self, dados):
        """Preserva ordem e FIPID; aceita categorias novas pelo OneHotEncoder."""
        if not dados[ID].is_unique or dados[ID].map(normalizar).isin(AUSENTES).any():
            raise ValueError("FIPID precisa ser preenchido e único para alinhamento.")
        x = construir_atributos(dados, self.config)[self.colunas_]
        saida = dados[[ID]].copy()
        saida["previsao"] = self.pipeline_.predict(x)
        if hasattr(self.pipeline_, "predict_proba"):
            saida["probabilidade_classe_1_nao_calibrada"] = self.pipeline_.predict_proba(x)[:, 1]
        else:
            saida["margem_classe_1"] = self.pipeline_.decision_function(x)
        return saida


def carregar_componente(caminho):
    """Carrega modelos confiáveis, inclusive os salvos antes da organização em pacotes."""
    import sys
    sys.modules.setdefault('clinica_pipeline', sys.modules[__name__])
    return joblib.load(caminho)


def iniciar_execucao(config):
    """Valida configuração, carrega CSV e registra proveniência em uma pasta nova."""
    if config.folds < 2 or config.permutacoes_associacao < 1 or config.repeticoes_importancia < 1:
        raise ValueError("Use ao menos 2 folds, 1 permutação e 1 repetição de importância.")
    dados = carregar_dados(config)
    pasta = Path(config.resultados) / datetime.now(timezone.utc).strftime("execucao_%Y%m%dT%H%M%S_%fZ")
    pasta.mkdir(parents=True, exist_ok=False)
    versoes = {p: importlib.metadata.version(p) for p in ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "joblib"]}
    metadados = {"configuracao": asdict(config), "versoes": versoes, "python": platform.python_version(),
        "csv_sha256": hashlib.sha256(Path(config.arquivo).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arquivo_absoluto": str(Path(config.arquivo).resolve()),
        "experimentos": EXPERIMENTOS, "alvo": {"0": ROTULOS[0], "1": ROTULOS[1]},
        "validacao": "StratifiedGroupKFold por Home number ID, shuffle=False; sem busca de hiperparâmetros"}
    (pasta / "configuracao_versoes.json").write_text(json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8")
    return dados, pasta


def finalizar_execucao(dados, coorte, x, resumo, metricas, associacoes, config, pasta):
    """Exporta componente de referência e relatório após concluir a validação."""
    # Referência pré-especificada, sem escolher o vencedor pela mesma validação.
    componente = ComponenteClinico(config).ajustar(coorte, coorte.alvo)
    joblib.dump(componente, pasta / "componente_demografico_logistica.joblib")
    texto = gerar_relatorio(dados, coorte, x, resumo, metricas, associacoes, pasta)
    (pasta / "RELATORIO.md").write_text(texto, encoding="utf-8")


def executar(config=None):
    """Executa etapas e cria uma pasta nova por execução, preservando resultados anteriores."""
    config = config or Configuracao()
    dados, pasta = iniciar_execucao(config)
    auditar_dados(dados, pasta)
    coorte = criar_coorte(dados, pasta)
    x = construir_atributos(coorte, config, pasta)
    analisar_coorte(coorte, x, pasta)
    associacoes = associacoes_por_casa(coorte, x, config, pasta)
    resumo, previsoes, metricas = avaliar_modelos(coorte, x, config, pasta)
    finalizar_execucao(dados, coorte, x, resumo, metricas, associacoes, config, pasta)
    print(f"Resultados: {pasta.resolve()}", flush=True)
    return {"pasta": pasta, "coorte": coorte, "atributos": x, "resumo": resumo,
            "previsoes": previsoes, "associacoes": associacoes}


def gerar_relatorio(dados, coorte, x, resumo, metricas, associacoes, pasta):
    """Redige somente números calculados, distinguindo hipóteses e limitações."""
    linhas = ["# Análise clínica — relatório calculado", "", "## Resultados calculados", "",
        f"Entrada: {len(dados)} registros, {len(COLUNAS)} colunas. Excluídos conforme rótulo original: {len(dados)-len(coorte)}.",
        f"Coorte: {len(coorte)} animais, {coorte[CASA].nunique()} casas; "
        f"{coorte.alvo.eq(0).sum()} Healthy housemate e {coorte.alvo.eq(1).sum()} casos incluídos.",
        f"Idade válida em {x.idade_anos.notna().sum()} animais. "
        f"Validação em {metricas.fold.nunique()} folds, sem compartilhamento de casas.", "",
        "Classe positiva: caso incluído no estudo. Desvio-padrão amostral entre folds; não é intervalo de confiança.", "",
        "|Experimento|Modelo|Accuracy|Precision|Recall|Especificidade|F1|ROC-AUC|", "|---|---|---|---|---|---|---|---|"]
    for (exp, modelo), linha in resumo.iterrows():
        valores = [f"{linha[m+'_media']:.3f} ± {linha[m+'_dp']:.3f}" for m in METRICAS]
        linhas.append("|"+"|".join([exp, modelo]+valores)+"|")
    linhas += ["", "No experimento demográfico, os resultados calculados de ROC-AUC foram: " +
        "; ".join(f"{nome}: {linha.roc_auc_media:.3f} ± {linha.roc_auc_dp:.3f}"
                  for nome, linha in resumo.loc["demografico"].iterrows()) + ". "
        "As diferenças observadas não estabelecem superioridade estatística entre os modelos.",
        "Ausência de forma clínica por grupo: " +
        "; ".join(f"{ROTULOS[c]}: {x.loc[coorte.alvo.eq(c), 'forma_clinica'].isna().sum()}/"
                  f"{coorte.alvo.eq(c).sum()}" for c in [0, 1]) + ". "
        "Esse padrão deve ser comparado às métricas dos experimentos ampliados antes de atribuir valor preditivo clínico."]
    linhas += ["", "## Interpretação e hipóteses", "",
        "O experimento demográfico avalia discriminação com idade, sexo e raça. "
        "Os demais medem discriminação retrospectiva com informação do processo diagnóstico. "
        "A comparação entre esses conjuntos não demonstra ganho clinicamente disponível antes da investigação.",
        "Forma clínica ausente/não aplicável, realização de exames e tratamento podem revelar como os grupos foram selecionados. "
        "Desempenho elevado nesses experimentos é compatível com esse viés; não comprova predição prospectiva.",
        "Associações usam permutações de rótulos dentro da casa, com correção Benjamini–Hochberg na família de testes exportada. "
        "A permutabilidade intradomiciliar é uma hipótese; casas sem contraste não informam o teste. "
        "V de Cramér e diferença média são efeitos descritivos, sem interpretação causal.",
        "A importância por permutação foi calculada nos folds de validação, nas variáveis originais. "
        "Quedas negativas de AUC são possíveis. Correlações e amostras pequenas tornam as importâncias instáveis; "
        "coeficientes por categoria codificada estão em arquivo separado e não são biomarcadores.", "",
        "## Limitações", "",
        "Não há documentação externa de confirmação diagnóstica nesta pasta. Very likely, Highly suspicious e Suspicious "
        "são graus de suspeita. Este alvo não mede PIF confirmada, prognóstico ou virulência. "
        "A grafia Very Likely é normalizada sem descartar o texto original.",
        "Amostra pequena, recrutamento selecionado, ausência diferencial e possível correlação familiar/doméstica limitam generalização. "
        "FIPID único não garante que dois IDs não correspondam ao mesmo animal; a observação sobre recoleta no rótulo de exclusão deve ser conferida na documentação do estudo.",
        "NA pode significar não aplicável, especialmente em controles; não é negativo nem ausência de doença. "
        "Sexo é mantido como código (MN/FS/MI/FI), sem presumir tradução não documentada. "
        "Idades dependem da precisão das datas de nascimento; o limite de 30 anos é configurável para auditoria.",
        "Não houve otimização de hiperparâmetros. Comparar e depois escolher o maior resultado nesta mesma CV produz otimismo de seleção. "
        "Não existe teste externo nem demonstração de utilidade clínica. Probabilidades não foram calibradas; SVM produz margens. "
        "Precision/F1 indefinidas por ausência de positivos previstos recebem zero explicitamente. "
        "AUC é calculada por fold, sem agregar margens SVM de modelos diferentes em uma única AUC.",
        "O componente serializado é a regressão logística demográfica pré-especificada, ajustada na coorte completa para reutilização; "
        "não há métrica de validação independente desse ajuste final. Carregar joblib somente de origem confiável.", "",
        "## Rastreabilidade e evolução futura", "",
        "Consulte configuracao_versoes.json, divisao_folds.csv, fluxo_inclusao.csv, atributos_derivados.csv e previsoes_oof.csv. "
        "Identificadores e grupo nunca entram como preditores. O CSV original permanece intacto.",
        "Para outras modalidades serão necessários ID de animal/amostra validado, tabela de correspondência entre amostras, "
        "casa, data e momento clínico, proveniência, critérios diagnósticos, disponibilidade de cada variável no momento da previsão, "
        "consentimento e partições de validação comuns. As outras modalidades são executadas independentemente por executar_tcc.py.", "",
        "Referências técnicas: https://scikit-learn.org/stable/modules/cross_validation.html ; "
        "https://scikit-learn.org/stable/modules/permutation_importance.html"]
    return "\n".join(linhas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arquivo", default=ARQUIVO_PADRAO)
    parser.add_argument("--resultados", default="resultados_clinicos")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--semente", type=int, default=42)
    args = parser.parse_args()
    # Importação pelo nome do módulo mantém o artefato joblib portável entre CLI e notebook.
    from clinica.clinica_pipeline import Configuracao, executar
    executar(Configuracao(**vars(args)))
