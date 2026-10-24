"""Avaliação exploratória independente da tabela proteômica fornecida."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from clinica.clinica_pipeline import Configuracao, criar_modelos
from avaliacao_comum import METRICAS, calcular_metricas
import matplotlib.pyplot as plt

ARQUIVO = 'JPST003448_all/21905517_merge-Metaboanalyst.txt'


class SelecionarTreino(BaseEstimator, TransformerMixin):
    """Filtra ausência e constantes e limita a 10 atributos somente no treino."""
    def fit(self, x, y):
        self.manter_ = np.mean(np.isnan(x), axis=0) <= .5
        if not self.manter_.any():
            raise ValueError('Nenhuma proteína com pelo menos 50% de observações no treino.')
        self.imputador_ = SimpleImputer(strategy='median')
        z = self.imputador_.fit_transform(x[:, self.manter_])
        self.variancia_ = VarianceThreshold().fit(z)
        z = self.variancia_.transform(z)
        # K fixado antes da avaliação; não otimizar na validação externa.
        self.seletor_ = SelectKBest(f_classif, k=min(10, z.shape[1])).fit(z, y)
        self.indices_ = np.flatnonzero(self.manter_)[self.variancia_.get_support()][self.seletor_.get_support()]
        return self

    def transform(self, x):
        z = self.imputador_.transform(x[:, self.manter_])
        return self.seletor_.transform(self.variancia_.transform(z))


def carregar_tabela(arquivo):
    """Usa apenas colunas com rótulo explícito; nenhuma inferência pelo nome."""
    d = pd.read_csv(arquivo, sep='\t', dtype=str, keep_default_na=False)
    if d.columns[0] != 'Protein' or d.Protein.eq('Label').sum() != 1:
        raise ValueError('Esquema/rótulo Label inválido.')
    rotulos = d.loc[d.Protein.eq('Label')].iloc[0]
    colunas = rotulos.index[rotulos.isin(['FIP', 'Healthy'])].tolist()
    if len(colunas) < 4:
        raise ValueError('Amostras rotuladas insuficientes.')
    corpo = d.loc[~d.Protein.eq('Label')]
    vazias = corpo.apply(lambda c: c.str.strip().eq('')).all(axis=1)
    corpo = corpo.loc[~vazias]
    if corpo.Protein.str.strip().eq('').any() or corpo.Protein.duplicated().any():
        raise ValueError('Proteínas sem ID ou duplicadas: requer política de agregação documentada.')
    x = corpo[colunas].replace('', np.nan).apply(pd.to_numeric, errors='raise').T
    x.columns = corpo.Protein.to_numpy()
    if np.isinf(x.to_numpy()).any() or (x < 0).any().any():
        raise ValueError('Valores infinitos/negativos: revisar escala antes da análise.')
    y = rotulos[colunas].map({'Healthy': 0, 'FIP': 1}).astype(int)
    return x, y, int(vazias.sum())


def inventariar(pasta, destino):
    """Registra corridas e QC estrutural; corridas não equivalem a animais."""
    linhas = []
    for p in sorted(Path(pasta).rglob('*')):
        if not p.is_file():
            continue
        linha = dict(arquivo=str(p), formato=p.suffix, bytes=p.stat().st_size)
        if p.suffix.lower() == '.mgf':
            inicio = fim = 0
            with p.open(errors='replace') as f:
                for s in f:
                    inicio += s.strip() == 'BEGIN IONS'
                    fim += s.strip() == 'END IONS'
            linha.update(espectros_inicio=inicio, espectros_fim=fim, estrutura_ok=inicio == fim and inicio > 0)
        elif p.suffix.lower() == '.mzxml':
            try:
                n = 0
                declarado = None
                for evento, elemento in ET.iterparse(p, events=('end',)):
                    tag = elemento.tag.split('}')[-1]
                    if tag == 'scan':
                        n += 1
                    if tag == 'msRun':
                        declarado = elemento.attrib.get('scanCount')
                    elemento.clear()
                linha.update(scans=n, scans_declarados=declarado, estrutura_ok=n > 0 and (declarado is None or n == int(declarado)))
            except (ET.ParseError, ValueError) as erro:
                linha.update(estrutura_ok=False, erro=str(erro))
        linhas.append(linha)
    pd.DataFrame(linhas).to_csv(destino / 'inventario_arquivos.csv', index=False)


def executar(arquivo=ARQUIVO, resultados='resultados_proteomicos'):
    x, y, vazias = carregar_tabela(arquivo)
    pasta = Path(resultados) / datetime.now(timezone.utc).strftime('execucao_%Y%m%dT%H%M%S_%fZ')
    pasta.mkdir(parents=True, exist_ok=False)
    inventariar(Path(arquivo).parent, pasta)
    x.to_csv(pasta / 'matriz_amostras_proteinas.csv')
    pd.DataFrame({'amostra': x.index, 'rotulo_original': y.map({0:'Healthy', 1:'FIP'}), 'alvo': y.to_numpy()}).to_csv(pasta / 'rotulos.csv', index=False)
    pd.DataFrame({'zeros': x.eq(0).sum(axis=1), 'ausentes': x.isna().sum(axis=1)}).to_csv(pasta / 'qc_amostras.csv')
    # Três folds fixos: somente dois controles por validação na base atual.
    if y.value_counts().min() < 3:
        raise ValueError('Menos de 3 observações na menor classe; avaliação suspensa.')
    folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(x, y))
    registros, previsoes, selecionadas, divisoes = [], [], [], []
    for fold, (tr, va) in enumerate(folds, 1):
        divisoes.extend(dict(amostra=x.index[i], fold=fold, alvo=int(y.iloc[i])) for i in va)
    pd.DataFrame(divisoes).to_csv(pasta / 'divisao_folds.csv', index=False)
    for politica in ['zeros_mantidos', 'zeros_ausentes']:
        # Zero como ausente é hipótese de sensibilidade, não fato sobre o ensaio.
        valores = x.mask(x.eq(0)) if politica == 'zeros_ausentes' else x
        a = valores.to_numpy(dtype=float)
        for nome, modelo in criar_modelos(Configuracao()).items():
            reais, preditos = [], []
            for fold, (tr, va) in enumerate(folds, 1):
                pipe = Pipeline([('selecao', SelecionarTreino()), ('escala', StandardScaler()), ('modelo', clone(modelo))])
                pipe.fit(a[tr], y.iloc[tr])
                pred = pipe.predict(a[va])
                escore = pipe.predict_proba(a[va])[:, 1] if hasattr(pipe, 'predict_proba') else pipe.decision_function(a[va])
                real = y.iloc[va]
                registros.append(dict(experimento=politica, modelo=nome, fold=fold, n_treino=len(tr), n_validacao=len(va),
                    **calcular_metricas(real, pred, escore)))
                selecionadas.extend(dict(experimento=politica, modelo=nome, fold=fold, proteina=x.columns[i]) for i in pipe['selecao'].indices_)
                previsoes.extend(dict(experimento=politica, modelo=nome, fold=fold, amostra=x.index[i], alvo=int(y.iloc[i]), previsao=int(p), escore=float(s), tipo_escore='margem' if nome=='SVM' else 'probabilidade_nao_calibrada') for i,p,s in zip(va,pred,escore))
                reais.extend(real); preditos.extend(pred)
            pd.DataFrame(confusion_matrix(reais,preditos,labels=[0,1]), index=['Healthy','FIP'],columns=['Healthy','FIP']).to_csv(pasta / f'matriz_{politica}_{nome}.csv')
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(reais, preditos, labels=[0, 1], display_labels=['Healthy', 'FIP'], ax=ax, colorbar=False)
            ax.set_title(f'Pools: {politica} / {nome}')
            fig.tight_layout()
            fig.savefig(pasta / f'matriz_{politica}_{nome}.png', dpi=150)
            plt.close(fig)
    m = pd.DataFrame(registros)
    m.to_csv(pasta / 'metricas_folds.csv', index=False)
    resumo = m.groupby(['experimento','modelo'])[METRICAS].agg(['mean','std'])
    resumo.columns = [f'{a}_{"media" if b=="mean" else "dp"}' for a,b in resumo.columns]
    resumo.to_csv(pasta / 'comparacao_modelos.csv')
    pd.DataFrame(previsoes).to_csv(pasta / 'previsoes_oof.csv', index=False)
    pd.DataFrame(selecionadas).to_csv(pasta / 'atributos_por_fold.csv', index=False)
    import importlib.metadata
    config = dict(arquivo=str(Path(arquivo).resolve()), sha256=hashlib.sha256(Path(arquivo).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), n_amostras=len(x), n_proteinas=x.shape[1],
        linhas_vazias_removidas=vazias, zeros=int(x.eq(0).sum().sum()), ausentes=int(x.isna().sum().sum()),
        classes=y.value_counts().to_dict(), unidade='pool/subgrupo; NÃO gato individual',
        fonte='https://www.nature.com/articles/s41598-025-03108-2',
        status='Demonstração computacional em pools pré-selecionados por perfil espectral; não valida diagnóstico individual',
        validacao='StratifiedKFold 3 folds shuffle=True random_state=42 sobre pools; seleção prévia não pode ser desfeita pela CV',
        transformacao='Escala fornecida, sem novo log ou normalização global; StandardScaler no treino',
        selecao='ausência <=50%, mediana, variância >0, ANOVA top min(10,p), tudo no treino',
        modelos={n:m.get_params() for n,m in criar_modelos(Configuracao()).items()},
        versoes={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy']})
    (pasta / 'configuracao.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    (pasta / 'RELATORIO.md').write_text(
        '# Proteômica — demonstração em pools\n\n' +
        f'{len(x)} pools, {x.shape[1]} atributos anotados pelo depositante; classes: {y.value_counts().to_dict()}.\n\n' +
        'Tabela quantitativa fornecida; MGF/mzXML usados somente para inventário estrutural. '
        'Mantidas duas hipóteses para zeros, sem escolher pela melhor métrica. '
        'Seleção, imputação e escala ajustadas no treino. Três folds sobre pools; '
        'independência entre animais não verificável e seleção anterior não refeita na CV. '
        'As métricas não validam diagnóstico individual. Consulte comparacao_modelos.csv, '
        'metricas_folds.csv, previsoes_oof.csv e matrizes OOF.\n', encoding='utf-8')
    print(f'Resultados proteômicos PROVISÓRIOS: {pasta.resolve()}',flush=True)
    return pasta


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arquivo',default=ARQUIVO)
    p.add_argument('--resultados',default='resultados_proteomicos')
    executar(**vars(p.parse_args()))
