"""Reúne métricas independentes sem ranking ou pareamento entre modalidades."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import json
from avaliacao_comum import METRICAS


def comparar(clinica, proteomica, resultados='resultados_comparacao', genomica=None):
    linhas = []
    for modalidade, caminho, alvo, estrategia, status in [
        ('clinica', clinica, 'Include por suspeita vs Healthy housemate; sem confirmação', 'StratifiedGroupKFold por casa', 'Exploratório; demográfico é referência'),
        ('proteomica', proteomica, 'Label FIP vs Healthy; suspeita clínica; maioria sem IHC segundo artigo', 'StratifiedKFold 3 sobre pools pré-selecionados por perfil espectral', 'Demonstração em pools; não estima diagnóstico individual')]:
        pasta = Path(caminho)
        m = pd.read_csv(pasta / 'metricas_folds.csv')
        resumo = pd.read_csv(pasta / 'comparacao_modelos.csv')
        oof = pd.read_csv(pasta / 'previsoes_oof.csv')
        id_col = 'FIPID' if modalidade == 'clinica' else 'amostra'
        for _, linha in resumo.iterrows():
            sel = (m.experimento == linha.experimento) & (m.modelo == linha.modelo)
            grupo = oof[(oof.experimento == linha.experimento) & (oof.modelo == linha.modelo)]
            linhas.append(dict(linha, modalidade=modalidade, n_amostras=grupo[id_col].nunique(), unidade='animal/registro' if modalidade=='clinica' else 'pool/subgrupo, não animal',
                n_classe_0=int(grupo.alvo.eq(0).sum()), n_classe_1=int(grupo.alvo.eq(1).sum()),
                folds=m.loc[sel,'fold'].nunique(), alvo=alvo, validacao=estrategia, status=status, origem=str(pasta.resolve())))
    destino = Path(resultados) / datetime.now(timezone.utc).strftime('execucao_%Y%m%dT%H%M%S_%fZ')
    destino.mkdir(parents=True,exist_ok=False)
    completa = pd.DataFrame(linhas)
    if genomica is not None:
        contagens = json.loads((Path(genomica) / 'contagens.json').read_text(encoding='utf-8'))
        linha_genomica = dict(modalidade='genomica', modelo='Indisponível', experimento='exploratorio',
            n_amostras=contagens['sequencias'], unidade='sequência; gatos independentes desconhecidos',
            n_classe_0=None, n_classe_1=None, folds=0, alvo='Nenhum alvo clínico validado disponível',
            validacao='Não realizada', status=contagens['motivo'], origem=str(Path(genomica).resolve()))
        for metrica in METRICAS:
            linha_genomica[f'{metrica}_media'] = float('nan')
            linha_genomica[f'{metrica}_dp'] = float('nan')
        completa = pd.concat([completa, pd.DataFrame([linha_genomica])], ignore_index=True)
    completa.to_csv(destino / 'comparacao_independente.csv',index=False, encoding='utf-8-sig', na_rep='Indisponível')
    # Referência definida antes de executar: não escolhe modelo/hipótese por desempenho.
    referencia = completa[(completa.modelo.eq('Regressao_logistica') & (
        (completa.modalidade.eq('clinica') & completa.experimento.eq('demografico')) |
        (completa.modalidade.eq('proteomica') & completa.experimento.eq('zeros_mantidos')))) |
        completa.modalidade.eq('genomica')]
    referencia.to_csv(destino / 'comparacao_por_modalidade.csv', index=False, encoding='utf-8-sig', na_rep='Indisponível')
    relatorio = ['# Comparação descritiva final', '',
        'Referência pré-especificada: regressão logística, clínica demográfica e proteômica com zeros mantidos. '
        'A hipótese alternativa de zeros e todos os modelos estão na tabela completa; não houve seleção de vencedor.', '',
        '|Modalidade|Modelo|F1|Precisão|Sensibilidade|Especificidade|ROC-AUC|',
        '|---|---|---|---|---|---|---|']
    for _, linha in referencia.iterrows():
        valores = [f'{linha[m+"_media"]:.3f} ± {linha[m+"_dp"]:.3f}'
                   if pd.notna(linha[m+'_media']) else 'Indisponível' for m in ['f1', 'precision', 'recall', 'specificity', 'roc_auc']]
        relatorio.append('|' + '|'.join([linha.modalidade, linha.modelo, *valores]) + '|')
    relatorio += ['', 'Médias não ponderadas e desvios amostrais entre folds; não são intervalos de confiança.',
        'Nenhum pareamento ou fusão. Populações, definições de diagnóstico e unidades diferem; não há ranking válido entre modalidades.',
        'Genômica: rótulos inferidos não são alvo clínico validado; métricas indisponíveis, sem classificador.',
        'Clínica: suspeita versus controle informado pelo estudo. Proteômica: demonstração em pools previamente selecionados.',
        '', 'Origens:']
    relatorio.extend(f'- {l.modalidade}: {l.origem}' for _, l in referencia.iterrows())
    (destino / 'RELATORIO.md').write_text('\n'.join(relatorio) + '\n', encoding='utf-8')
    (destino / 'LIMITACOES.txt').write_text('Comparação descritiva entre bases diferentes. Não pareadas. Rótulos, população, coleta, tamanho e validação diferem. A comparação descritiva não demonstra superioridade diagnóstica. DP entre folds não é intervalo de confiança. Experimentos clínicos ampliados têm vazamento semântico. Proteômica: 17 pools/subgrupos pré-selecionados por perfil espectral, não 17 animais. A CV não desfaz seleção anterior e não estima desempenho diagnóstico individual. Fonte: https://www.nature.com/articles/s41598-025-03108-2',encoding='utf-8')
    print(destino.resolve())
    return destino


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clinica',required=True)
    p.add_argument('--proteomica',required=True)
    p.add_argument('--resultados',default='resultados_comparacao')
    p.add_argument('--genomica')
    comparar(**vars(p.parse_args()))
