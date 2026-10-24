"""Consolida artefatos de uma única execução identificada por manifesto e hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from avaliacao_comum import METRICAS

NOMES = {'clinica': 'Clínica', 'proteomica': 'Proteômica', 'genomica': 'Genômica'}
REQUERIDOS = {
    'clinica': ['configuracao_versoes.json', 'fluxo_inclusao.csv', 'atributos_derivados.csv',
                'comparacao_modelos.csv', 'metricas_folds.csv', 'previsoes_oof.csv', 'divisao_folds.csv'],
    'proteomica': ['configuracao.json', 'matriz_amostras_proteinas.csv', 'rotulos.csv',
                   'comparacao_modelos.csv', 'metricas_folds.csv', 'previsoes_oof.csv',
                   'atributos_por_fold.csv', 'inventario_arquivos.csv'],
    'genomica': ['contagens.json', 'configuracao_versoes.json', 'auditoria_genomica.csv',
                 'features_genomicas.csv', 'metadados_integrados.csv', 'validacao_arquivos.csv'],
}


def hash_arquivo(caminho):
    """Calcula SHA-256 de um artefato técnico."""
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def caminho_da_etapa(execucao, etapa):
    """Resolve caminho relativo ou legado, impedindo mistura entre execuções."""
    pasta = Path(etapa['pasta'])
    pasta = pasta.resolve() if pasta.is_absolute() else (Path(execucao) / pasta).resolve()
    if not pasta.is_relative_to(Path(execucao).resolve()):
        raise ValueError('Artefato fora da execução indicada; mistura de execuções rejeitada.')
    return pasta


def tabela_md(colunas, linhas):
    """Formata Markdown sem dependência adicional de tabulate."""
    def limpar(valor):
        return str(valor).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['|' + '|'.join(colunas) + '|', '|' + '|'.join(['---'] * len(colunas)) + '|',
                      *('|' + '|'.join(limpar(v) for v in linha) + '|' for linha in linhas)])


def medida(linha, nome):
    """Nunca converte métricas ausentes em zero."""
    if linha is None or pd.isna(linha.get(nome + '_media')):
        return 'N/A'
    media, dp = linha[nome + '_media'], linha.get(nome + '_dp', np.nan)
    return f'{media:.3f} ± {dp:.3f}' if pd.notna(dp) else f'{media:.3f} (DP N/A)'


def carregar_execucao(execucao):
    """Usa somente etapas concluídas e artefatos íntegros do manifesto informado."""
    execucao = Path(execucao).resolve()
    manifesto = json.loads((execucao / 'manifesto_execucao.json').read_text(encoding='utf-8'))
    if manifesto.get('execucao_id') != execucao.name:
        raise ValueError('Manifesto sem identificador compatível; use uma execução com o novo esquema.')
    dados = {}
    for nome in NOMES:
        etapa = manifesto.get('etapas', {}).get(nome, {})
        resultado = dict(status='Não executada', motivo=etapa.get('motivo', 'Etapa ausente no manifesto'), pasta=None)
        if etapa.get('status') == 'falhou':
            resultado.update(status='Falhou; possíveis artefatos parciais não usados', motivo=etapa.get('erro', 'Erro não informado'))
        elif etapa.get('status') == 'concluida':
            try:
                pasta = caminho_da_etapa(execucao, etapa)
                hashes = etapa.get('artefatos_sha256', {})
                if not hashes:
                    raise ValueError('Etapa sem hashes de proveniência')
                for relativo, esperado in hashes.items():
                    arquivo = (pasta / relativo).resolve()
                    if not arquivo.is_relative_to(pasta) or hash_arquivo(arquivo) != esperado:
                        raise ValueError(f'Artefato alterado ou caminho incompatível: {relativo}')
                for requerido in REQUERIDOS[nome]:
                    if requerido not in hashes:
                        raise ValueError(f'Artefato obrigatório não registrado: {requerido}')
                resultado.update(status='Executada com sucesso', motivo='', pasta=pasta, hashes=hashes)
                for arquivo in REQUERIDOS[nome]:
                    caminho = pasta / arquivo
                    resultado[arquivo] = (json.loads(caminho.read_text(encoding='utf-8')) if arquivo.endswith('.json')
                                          else pd.read_csv(caminho, keep_default_na=False))
                if nome != 'genomica':
                    # As médias são conferidas contra os folds; não se copia texto de relatórios antigos.
                    resumo, folds = resultado['comparacao_modelos.csv'], resultado['metricas_folds.csv']
                    for m in METRICAS:
                        for sufixo in ['media', 'dp']:
                            resumo[m + '_' + sufixo] = pd.to_numeric(resumo[m + '_' + sufixo], errors='coerce')
                    for _, linha in resumo.iterrows():
                        grupo = folds[(folds.experimento == linha.experimento) & (folds.modelo == linha.modelo)]
                        if grupo.empty:
                            raise ValueError('Resumo sem avaliações de origem')
                        for m in METRICAS:
                            valores = pd.to_numeric(grupo[m], errors='coerce')
                            for sufixo, valor in [('media', valores.mean()), ('dp', valores.std())]:
                                observado = pd.to_numeric(linha[m + '_' + sufixo], errors='coerce')
                                if not np.isclose(observado, valor, equal_nan=True):
                                    raise ValueError(f'Resumo incoerente com folds: {m}')
                else:
                    resultado['motivo'] = resultado['contagens.json']['motivo']
                    resultado['status'] = 'Executada com sucesso; sem métricas supervisionadas'
            except (OSError, ValueError, KeyError, TypeError) as erro:
                resultado = dict(status='Parcial; artefatos ausentes ou inválidos', motivo=str(erro), pasta=None)
        dados[nome] = resultado
    return manifesto, dados


def construir_comparacao(manifesto, dados):
    """Uma linha por modalidade; referência já definida, sem selecionar maior escore."""
    linhas = []
    for nome, d in dados.items():
        linha = dict(execucao_id=manifesto['execucao_id'], modalidade=NOMES[nome], fonte='N/A',
            amostras_independentes='Não estabelecido', unidades_observadas='N/A', caracteristicas='N/A',
            classes='N/A', algoritmos='N/A', validacao='N/A', modelo_referencia='N/A',
            status=d['status'], observacoes=d['motivo'])
        referencia = None
        if d['pasta'] is not None and nome == 'clinica':
            fluxo = d['fluxo_inclusao.csv']
            incluido = fluxo.incluido_na_analise.astype(str).str.lower().eq('true')
            coorte = fluxo.loc[incluido]
            config = d['configuracao_versoes.json']
            ref = config['experimentos']['demografico']
            linha.update(fonte=manifesto['configuracao']['clinica'] + '; Dryad mencionado no nome, referência completa pendente',
                unidades_observadas=f'{len(coorte)} registros; {coorte["Home number ID"].nunique()} casas',
                caracteristicas=f'{len(ref)} variáveis originais; dimensão OneHot varia por treino',
                classes='Include por suspeita (1) vs Healthy housemate informado (0)',
                validacao=f'{d["metricas_folds.csv"].fold.nunique()} folds por casa',
                observacoes='Independência por gato não confirmada; suspeita não é diagnóstico definitivo')
        elif d['pasta'] is not None and nome == 'proteomica':
            config = d['configuracao.json']
            linha.update(fonte=manifesto['configuracao']['proteomica'] + '; JPST003448',
                unidades_observadas=f'{config["n_amostras"]} pools/subgrupos',
                caracteristicas=f'{config["n_proteinas"]} originais; até 10 selecionadas por treino',
                classes='FIP (1) vs Healthy (0), conforme Label',
                validacao=f'{d["metricas_folds.csv"].fold.nunique()} folds estratificados de pools',
                observacoes='Composição por gato desconhecida; seleção anterior à CV; referência com zeros mantidos')
        elif d['pasta'] is not None:
            contagens = d['contagens.json']
            config = d['configuracao_versoes.json']
            linha.update(fonte=manifesto['configuracao']['genomica'] + '; FASTA locais, origem clínica não documentada',
                unidades_observadas=f'{contagens["sequencias"]} sequências',
                caracteristicas=len(config['atributos_preditores']),
                classes=f'Rótulos iniciais {contagens["rotulos_iniciais"]}; não são alvo validado',
                algoritmos='Nenhum classificador; análise exploratória', validacao='Não realizada')
        if d['pasta'] is not None and nome != 'genomica':
            resumo = d['comparacao_modelos.csv']
            experimento = 'demografico' if nome == 'clinica' else 'zeros_mantidos'
            selecionado = resumo[resumo.modelo.eq('Regressao_logistica') & resumo.experimento.eq(experimento)]
            if len(selecionado) != 1:
                raise ValueError(f'Referência pré-especificada ausente em {nome}')
            referencia = selecionado.iloc[0]
            linha.update(algoritmos=', '.join(sorted(resumo.modelo.unique())), modelo_referencia=f'Regressao_logistica / {experimento}')
        for m in METRICAS:
            linha[m] = medida(referencia, m)
            linha[m + '_media'] = float(referencia[m + '_media']) if referencia is not None else np.nan
            linha[m + '_dp'] = float(referencia[m + '_dp']) if referencia is not None else np.nan
        linhas.append(linha)
    return pd.DataFrame(linhas)


def renderizar(manifesto, dados, comparacao, execucao, base):
    """Renderiza os mesmos valores com links relativos à localização do documento."""
    def link(arquivo, titulo=None, imagem=False, gerado=False):
        arquivo = Path(arquivo)
        if not arquivo.is_file() and not gerado:
            raise FileNotFoundError(f'Link para artefato inexistente: {arquivo}')
        relativo = quote(Path(os.path.relpath(arquivo, base)).as_posix(), safe='/._-')
        return f'{"!" if imagem else ""}[{titulo or arquivo.name}]({relativo})'

    def secao_metricas(d):
        resumo = d['comparacao_modelos.csv']
        return tabela_md(['Experimento', 'Modelo', 'Accuracy', 'Precisão', 'Sensibilidade', 'Especificidade', 'F1', 'ROC-AUC'],
                         ([l.experimento, l.modelo, *(medida(l, m) for m in METRICAS)] for _, l in resumo.iterrows()))

    def maior_observado(d, experimento):
        resumo = d['comparacao_modelos.csv']
        candidatos = resumo[resumo.experimento.eq(experimento) & resumo.modelo.ne('Dummy')]
        maior = candidatos.f1_media.max()
        nomes = candidatos.loc[np.isclose(candidatos.f1_media, maior), 'modelo'].tolist()
        return (f'Maior F1 observado em {experimento}: {", ".join(nomes)}, {maior:.3f}. '
                'Descrição retrospectiva, sem superioridade estatística demonstrada. '
                'Não foi definido critério de seleção de vencedor antes da análise; '
                'portanto não há “melhor modelo” validado. A referência comparativa continua sendo a regressão logística.')

    def matrizes(d):
        linhas = []
        for _, l in d['comparacao_modelos.csv'].iterrows():
            arquivo = d['pasta'] / f'matriz_{l.experimento}_{l.modelo}.csv'
            if arquivo.name not in d['hashes']:
                raise ValueError(f'Matriz não registrada no manifesto: {arquivo.name}')
            a = pd.read_csv(arquivo, index_col=0).to_numpy()
            if a.shape != (2, 2):
                raise ValueError('Matriz binária com dimensões incompatíveis')
            linhas.append([l.experimento, l.modelo, *a.ravel(), link(arquivo)])
        return tabela_md(['Experimento', 'Modelo', 'TN', 'FP', 'FN', 'TP', 'Origem'], linhas)

    linhas = ['# Relatório final integrado — classificação exploratória da PIF', '',
        f'Execução de origem: **{manifesto["execucao_id"]}**. Início UTC: {manifesto["iniciada_em_utc"]}. '
        f'Estado registrado do pipeline: **{manifesto["status"]}**.', '',
        '## 1. Resumo executivo', '',
        'Relatório construído a partir dos artefatos registrados e conferidos por SHA-256 desta execução. '
        'Nenhuma métrica foi importada de outra execução. N/A significa não calculado ou não aplicável, nunca zero.', '',
        tabela_md(['Modalidade', 'Situação', 'Motivo/limite'],
                  ([NOMES[n], d['status'], d['motivo'] or 'Concluída; interpretação exploratória'] for n, d in dados.items())), '',
        '## 2. Objetivo da análise', '',
        'Comparar modelos de Machine Learning baseados em dados clínicos, proteômicos e genômicos para a classificação '
        'da Peritonite Infecciosa Felina (PIF), explicitando o alcance exploratório permitido pelos dados.', '',
        '## 3. Dados utilizados e suas fontes', '',
        'As fontes são descritas a partir das configurações e da documentação local. A origem bibliográfica '
        'clínica completa e os estudos de cada genoma não estão documentados nos metadados; não foram inventados.', '',
        tabela_md(['Modalidade', 'Fonte', 'Unidades observadas'],
                  ([l.modalidade, l.fonte, l.unidades_observadas] for _, l in comparacao.iterrows())), '',
        '## 4. Metodologia geral', '',
        'Análises independentes, sem pareamento ou late fusion. Pré-processamento aprendido apenas nos treinos. '
        'Mantidos Dummy, regressão logística, SVM linear e Random Forest onde há avaliação supervisionada. '
        'Hiperparâmetros fixos, ponderação balanced nos modelos aprendidos, sem SMOTE ou busca de hiperparâmetros. '
        'Consulte as configurações vinculadas ao final para valores exatos e sementes.', '',
        'Métricas por fold: accuracy, precisão, recall/sensibilidade, especificidade, F1 e ROC-AUC. '
        'Média não ponderada ± desvio amostral entre folds; não é intervalo de confiança. '
        'AUC usa a margem SVM por fold, sem agregar margens de modelos distintos; demais escores não são calibrados. '
        'Precisão/F1 sem positivos previstos recebem zero conforme o protocolo; métricas não aplicáveis recebem N/A. '
        'Matrizes são OOF: linhas observadas e colunas previstas, ordem 0/1.', '',
        '## 5. Resultados da modalidade clínica', '']
    d = dados['clinica']
    if d['pasta'] is None:
        linhas += [f'{d["status"]}: {d["motivo"]}. Métricas N/A.']
    else:
        fluxo = d['fluxo_inclusao.csv']
        incluir = fluxo.incluido_na_analise.astype(str).str.lower().eq('true')
        coorte = fluxo.loc[incluir]
        grupos = coorte['Included/excluded and group'].value_counts().to_dict()
        config = d['configuracao_versoes.json']
        linhas += [f'Registros encontrados: {len(fluxo)}; incluídos: {len(coorte)}; excluídos: {int((~incluir).sum())}. '
            f'Casas: {coorte["Home number ID"].nunique()}. Distribuição dos grupos: {grupos}.', '',
            'Critério: somente Include e Healthy housemate; exclusões originais mantidas. '
            'Very likely, Highly suspicious e Suspicious são suspeitas, sem confirmação presumida. '
            'Referência: idade, sexo e raça. Idade por datas estritas mês/dia/ano; inválidas/futuras/implausíveis ficam ausentes. '
            'Mediana, StandardScaler e OneHot ajustados no treino. Experimentos com forma, exames e tratamento '
            'continuam exploratórios e sujeitos a vazamento semântico. Validação agrupada por casa.', '',
            'Distribuição das categorias diagnósticas entre incluídos:', '',
            tabela_md(['Categoria original', 'Registros'], coorte['Diagnosis Category'].value_counts().items()), '',
            'Características originais por experimento: ' + str({k: len(v) for k, v in config['experimentos'].items()}) + '.', '',
            secao_metricas(d), '', maior_observado(d, 'demografico'), '', 'Matrizes de confusão OOF:', '', matrizes(d), '']
        for arquivo in ['classes.png', 'idade_grupo.png', 'comparacao_demografico.png',
                        'matriz_demografico_Regressao_logistica.png', 'roc_demografico_Regressao_logistica.png']:
            linhas += [link(d['pasta'] / arquivo, imagem=True), '']
        linhas += ['A regressão logística serializada foi ajustada na coorte completa para reutilização, sem teste externo '
                   'desse ajuste. Importâncias e coeficientes são exploratórios, não biomarcadores.', '']
    linhas += ['', '## 6. Resultados da modalidade proteômica', '']
    d = dados['proteomica']
    if d['pasta'] is None:
        linhas += [f'{d["status"]}: {d["motivo"]}. Métricas N/A.']
    else:
        c = d['configuracao.json']
        linhas += [f'Fonte: JPST003448, tabela quantitativa depositada. {c["n_amostras"]} pools/subgrupos e '
            f'{c["n_proteinas"]} atributos anotados; classes {c["classes"]}. '
            f'Linhas totalmente vazias removidas: {c["linhas_vazias_removidas"]}; '
            f'zeros: {c["zeros"]}; ausentes literais: {c["ausentes"]}.', '',
            'A tabela foi transposta para pools nas linhas. MGF/mzXML só foram inventariados estruturalmente; '
            'não foram usados diretamente como abundâncias. Não foi repetida identificação proteica/FDR. '
            'Duas hipóteses pré-especificadas para zeros, ambas apresentadas; sem escolha pela maior métrica.', '',
            f'Transformação: {c["transformacao"]}. Seleção: {c["selecao"]}. Validação: {c["validacao"]}.', '',
            secao_metricas(d), '', maior_observado(d, 'zeros_mantidos'), '',
            'Matrizes de confusão OOF:', '', matrizes(d), '']
        for politica in ['zeros_mantidos', 'zeros_ausentes']:
            linhas += [link(d['pasta'] / f'matriz_{politica}_Regressao_logistica.png', imagem=True), '']
        selecao = d['atributos_por_fold.csv']
        selecao = selecao[selecao.experimento.eq('zeros_mantidos') & selecao.modelo.eq('Regressao_logistica')]
        frequencias = selecao.groupby('proteina').fold.nunique().sort_values(ascending=False).head(10)
        linhas += ['Atributos mais frequentemente selecionados na referência (até dez exibidos; frequência por fold, '
                   'sem interpretação como biomarcadores):', '', tabela_md(['Identificador', 'Folds'], frequencias.items()), '']
        inventario = d['inventario_arquivos.csv']
        ruins = inventario[inventario.estrutura_ok.astype(str).str.lower().eq('false')]
        linhas += [f'Inventário: {len(inventario)} arquivos, {len(ruins)} alertas estruturais. '
            'Não foram removidas colunas quantitativas com base nas corridas sem mapa validado para os pools.', '',
            'Pools foram pré-selecionados antes desta validação. A composição por gato é desconhecida. '
            'As métricas não representam diagnóstico individual validado.', '']
    linhas += ['', '## 7. Resultados da modalidade genômica', '']
    d = dados['genomica']
    if d['pasta'] is None:
        linhas += [f'{d["status"]}: {d["motivo"]}. Métricas N/A.']
    else:
        c, config = d['contagens.json'], d['configuracao_versoes.json']
        a, meta = d['auditoria_genomica.csv'], d['metadados_integrados.csv']
        desconhecidos = int(meta.estado_diagnostico.eq('desconhecido').sum())
        linhas += [f'FASTA locais de FCoV: {c["arquivos"]} arquivos, {c["sequencias"]} sequências, '
            f'{c["arquivos_com_multiplas_sequencias"]} arquivos multissequência, '
            f'{c["arquivos_com_problemas"]} arquivos com problemas e {c["sequencias_repetidas"]} sequências repetidas. '
            f'Gatos independentes: {c["gatos_independentes_confirmados"] if c["gatos_independentes_confirmados"] is not None else "não estabelecido"}.', '',
            f'Rótulos iniciais: {c["rotulos_iniciais"]}. Estados diagnósticos: {c["estados_diagnostico"]}. '
            f'Confirmados utilizáveis: {c["diagnosticos_confirmados"]}; estado desconhecido: {desconhecidos}; '
            f'alvos clínicos vazios nas sequências: {int(a.alvo_clinico_validado.eq("").sum())}. '
            'Rótulos provisórios inferidos não equivalem a diagnóstico conhecido.', '',
            'Confirmação estrutural exige estado confirmado, rotulo_validado 0/1 e referência preenchida; '
            'a veracidade clínica da referência precisa de revisão humana. cat_id e estudo_id não são inventados. '
            'Accessions são extraídos por formato, sem comprovar fonte clínica ou gatos distintos.', '',
            f'Características numéricas: {len(config["atributos_preditores"])}. Transformação: {config["transformacao"]}. '
            f'k={config["k"]}. Alinhadores encontrados: {config["alinhadores"]}. '
            'Sem alinhamento validado, referência selecionada ou anotação por gene; não há variantes posicionais. '
            'U e ambiguidades IUPAC são aceitos, mas somente janelas ACGT entram nas frequências; cobertura explícita. '
            'Nomes, accession, estudo e rótulo inicial não são preditores.', '',
            f'Algoritmos supervisionados avaliados: nenhum. Métricas: N/A. {c["motivo"]}. '
            f'Sequências não usadas no treino: {c["sequencias_nao_usadas_treinamento"]}, com motivo registrado por linha.', '',
            f'Descrições divergentes de nomes: {c["descricoes_divergentes"]}. '
            'Grupos potenciais por nome são alertas, não identificadores de gatos confirmados.', '',
            tabela_md(['Arquivo', 'Accession', 'Cabeçalho'],
                ([l.arquivo, l.accession, l.cabecalho] for _, l in a[a.descricao_diverge_nome.astype(str).str.lower().eq('true')].iterrows())), '',
            link(d['pasta'] / 'qualidade_genomica.png', imagem=True), '']
    linhas += ['', '## 8. Tabela comparativa entre as modalidades', '',
        'Referência pré-especificada: regressão logística clínica demográfica e proteômica com zeros mantidos. '
        'N/A genômico significa ausência de avaliação supervisionada justificável; em falhas significa ausência de artefatos válidos. '
        'Amostras independentes não são presumidas a partir de registros, pools ou sequências.', '',
        tabela_md(['Modalidade', 'Fonte', 'Amostras independentes', 'Unidades observadas', 'Características', 'Classes',
                   'Algoritmos', 'Validação', 'Referência', 'Precisão', 'Sensibilidade', 'Especificidade', 'F1', 'ROC-AUC', 'Observações'],
                  ([l.modalidade, l.fonte, l.amostras_independentes, l.unidades_observadas, l.caracteristicas,
                    l.classes, l.algoritmos, l.validacao, l.modelo_referencia,
                    *(l[m] for m in ['precision', 'recall', 'specificity', 'f1', 'roc_auc']), l.observacoes]
                   for _, l in comparacao.iterrows())), '',
        '## 9. Interpretação dos resultados', '',
        'Não há comparação controlada nos mesmos gatos. Populações, definições de diagnóstico, unidades, coleta e validação diferem. '
        'O maior número não determina superioridade de modalidade. Melhores escores observados dentro de um experimento são '
        'descrições da mesma CV, não seleção confirmada em teste externo. Experimentos clínicos ampliados podem recuperar o '
        'processo de seleção; a CV proteômica não desfaz o agrupamento/seleção anterior.', '',
        '## 10. Limitações dos dados e dos modelos', '',
        'Suspeitas clínicas e controles informados sem adjudicação; ausências diferenciais e possível recoleta. '
        'Pools sem composição individual; seleção espectral anterior e significado dos zeros pendente. '
        'Genomas com diagnóstico/estudo/animal não documentados, possíveis tecidos relacionados e divergências de nomes. '
        'Sem calibração, teste externo ou evidência de utilidade clínica. Falhas e resultados parciais são expostos acima '
        'e não recebem métricas de execuções anteriores.', '',
        '## 11. Conclusões', '',
        'As etapas válidas desta execução sustentam as análises exploratórias descritas. '
        'A classificação de PIF clinicamente confirmada e a comparação de superioridade entre modalidades '
        'exigem adjudicação diagnóstica, unidades independentes e validação externa. '
        'Ausência de dados não foi preenchida com resultados fictícios.', '',
        '## 12. Referências e fontes dos dados', '',
        '- Clínica: arquivo local FIP and Housemate Clinical Data; o nome menciona Dryad, mas DOI/artigo não estão documentados.',
        '- Proteômica: depósito JPST003448 e [artigo associado na documentação local](https://www.nature.com/articles/s41598-025-03108-2).',
        '- Genômica: FASTA e metadados locais; accessions e cabeçalhos disponíveis na auditoria. Estudos de origem precisam de preenchimento.',
        '- Referências técnicas já usadas no projeto: [validação cruzada](https://scikit-learn.org/stable/modules/cross_validation.html) '
        'e [importância por permutação](https://scikit-learn.org/stable/modules/permutation_importance.html).', '',
        '## 13. Arquivos complementares e caminhos dos artefatos gerados', '',
        link(Path(execucao) / 'manifesto_execucao.json', 'Manifesto da execução'), '',
        link(Path(execucao) / 'tabela_comparativa.csv', 'Tabela comparativa em CSV', gerado=True), '',
        'Relatórios individuais são complementos técnicos; este documento é a síntese principal. '
        'As execuções históricas foram preservadas e não são fontes desta síntese.', '']
    for nome, d in dados.items():
        if d['pasta'] is not None:
            linhas += [f'### {NOMES[nome]}', '']
            linhas.extend('- ' + link(d['pasta'] / arquivo) for arquivo in sorted(d['hashes']))
            linhas.append('')
    return '\n'.join(linhas) + '\n'


def gerar(execucao=None, resultados='resultados'):
    """Preserva síntese por execução e atualiza a principal sem buscar execuções antigas."""
    resultados = Path(resultados).resolve()
    if execucao is None:
        atual = json.loads((resultados / 'execucao_atual.json').read_text(encoding='utf-8'))
        execucao = (resultados / atual['execucao']).resolve()
        if execucao.name != atual['execucao_id']:
            raise ValueError('Identificador do apontador atual não corresponde à execução')
    execucao = Path(execucao).resolve()
    manifesto, dados = carregar_execucao(execucao)
    comparacao = construir_comparacao(manifesto, dados)
    csv = comparacao.to_csv(index=False, na_rep='N/A', lineterminator='\n')
    # A cópia por execução é imutável; regeneração idêntica é permitida.
    destinos = [(execucao / 'tabela_comparativa.csv', csv, 'utf-8-sig'),
                (execucao / 'relatorio_final_integrado.md', renderizar(manifesto, dados, comparacao, execucao, execucao), 'utf-8')]
    for caminho, conteudo, encoding in destinos:
        if caminho.exists() and caminho.read_text(encoding=encoding) != conteudo:
            raise FileExistsError(f'Síntese histórica diferente já existe: {caminho}; preserve-a e crie outra execução')
    resultados.mkdir(parents=True, exist_ok=True)
    for caminho, conteudo, encoding in destinos:
        if not caminho.exists():
            with caminho.open('x', encoding=encoding, newline='') as arquivo:
                arquivo.write(conteudo)
    # A versão principal aponta sempre para a execução explicitamente solicitada, mesmo incompleta.
    (resultados / 'tabela_comparativa.csv').write_text(csv, encoding='utf-8-sig', newline='')
    principal = resultados / 'relatorio_final_integrado.md'
    principal.write_text(renderizar(manifesto, dados, comparacao, execucao, resultados), encoding='utf-8', newline='')
    ponteiro = dict(execucao_id=manifesto['execucao_id'], execucao=Path(os.path.relpath(execucao, resultados)).as_posix(),
                    status=manifesto['status'])
    (resultados / 'execucao_atual.json').write_text(json.dumps(ponteiro, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Relatório integrado: {principal}', flush=True)
    return principal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execucao', help='Pasta com manifesto; padrão: apontador explícito execucao_atual.json')
    parser.add_argument('--resultados', default='resultados', help='Diretório do relatório principal')
    gerar(**vars(parser.parse_args()))
