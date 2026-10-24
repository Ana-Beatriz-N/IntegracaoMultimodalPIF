"""Confere integridade das saídas reais de uma execução completa do TCC."""
import argparse
import hashlib
import json
import re
from urllib.parse import unquote
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from gerar_relatorio_integrado import caminho_da_etapa


def verificar(execucao, inventario=None):
    """Recalcula métricas OOF e verifica unidade, grupos e preservação de entradas."""
    execucao = Path(execucao)
    manifesto = json.loads((execucao / 'manifesto_execucao.json').read_text(encoding='utf-8'))
    if manifesto['status'] != 'concluida':
        raise ValueError('Manifesto não declara execução completa; consultar logs.')
    evidencias = {}
    for modalidade, identificador in [('clinica', 'FIPID'), ('proteomica', 'amostra')]:
        pasta = caminho_da_etapa(execucao, manifesto['etapas'][modalidade])
        oof = pd.read_csv(pasta / 'previsoes_oof.csv')
        metricas = pd.read_csv(pasta / 'metricas_folds.csv')
        divisao = pd.read_csv(pasta / 'divisao_folds.csv')
        assert not oof.duplicated(['experimento', 'modelo', identificador]).any()
        assert not divisao[identificador].duplicated().any()
        if modalidade == 'clinica':
            assert divisao.groupby('Home number ID').fold.nunique().max() == 1
        for _, linha in metricas.iterrows():
            grupo = oof[(oof.experimento == linha.experimento) &
                        (oof.modelo == linha.modelo) & (oof.fold == linha.fold)]
            assert len(grupo) == linha.n_validacao
            tn, fp, fn, tp = confusion_matrix(grupo.alvo, grupo.previsao, labels=[0, 1]).ravel()
            np.testing.assert_allclose(linha.specificity, tn / (tn + fp))
            np.testing.assert_allclose(linha.recall, tp / (tp + fn))
            np.testing.assert_allclose(linha.roc_auc, roc_auc_score(grupo.alvo, grupo.escore))
        for (experimento, modelo), grupo in oof.groupby(['experimento', 'modelo']):
            assert set(grupo[identificador]) == set(divisao[identificador])
            matriz = pd.read_csv(pasta / f'matriz_{experimento}_{modelo}.csv', index_col=0).to_numpy()
            np.testing.assert_array_equal(matriz, confusion_matrix(grupo.alvo, grupo.previsao, labels=[0, 1]))
        evidencias[modalidade] = dict(previsoes_oof=len(oof), avaliacoes_por_fold=len(metricas),
                                     unidades=int(oof[identificador].nunique()))
    pasta = caminho_da_etapa(execucao, manifesto['etapas']['genomica'])
    auditoria = pd.read_csv(pasta / 'auditoria_genomica.csv', keep_default_na=False)
    matriz = pd.read_csv(pasta / 'matriz_genomica_exploratoria.csv', keep_default_na=False)
    features = pd.read_csv(pasta / 'features_genomicas.csv')
    assert auditoria.amostra_id.is_unique and len(matriz) == len(auditoria)
    assert matriz.alvo_clinico_validado.eq('').all(), 'Verificação desta base espera ausência de diagnóstico confirmado'
    np.testing.assert_allclose(features.filter(like='kmer_').sum(axis=1), 1)
    assert not (pasta / 'metricas_folds.csv').exists()
    comparacao = caminho_da_etapa(execucao, manifesto['etapas']['comparacao'])
    referencia = pd.read_csv(comparacao / 'comparacao_por_modalidade.csv', keep_default_na=False)
    completa = pd.read_csv(comparacao / 'comparacao_independente.csv', keep_default_na=False)
    assert len(referencia) == 3 and set(referencia.modalidade) == {'clinica', 'proteomica', 'genomica'}
    assert referencia.loc[referencia.modalidade.eq('genomica'), 'f1_media'].iloc[0] == 'Indisponível'
    evidencias['genomica'] = dict(sequencias=len(auditoria), diagnosticos_confirmados=0, treinamento=False)
    evidencias['comparacao'] = dict(linhas_referencia=len(referencia), linhas_detalhadas=len(completa))
    if 'execucao_id' in manifesto:
        # A referência é a entrada lida nesta execução, não um inventário de outra versão dos dados.
        clinica = caminho_da_etapa(execucao, manifesto['etapas']['clinica'])
        proteomica = caminho_da_etapa(execucao, manifesto['etapas']['proteomica'])
        genomica = caminho_da_etapa(execucao, manifesto['etapas']['genomica'])
        c = json.loads((clinica / 'configuracao_versoes.json').read_text(encoding='utf-8'))
        p = json.loads((proteomica / 'configuracao.json').read_text(encoding='utf-8'))
        g = json.loads((genomica / 'configuracao_versoes.json').read_text(encoding='utf-8'))
        entradas_atuais = {manifesto['configuracao']['clinica']: c['csv_sha256'],
                          manifesto['configuracao']['proteomica']: p['sha256'], **g['entradas_sha256']}
        for caminho, esperado in entradas_atuais.items():
            assert hashlib.sha256(Path(caminho).read_bytes()).hexdigest() == esperado, caminho
        evidencias['entradas_da_execucao_conferidas'] = len(entradas_atuais)
        integrada = pd.read_csv(execucao / 'tabela_comparativa.csv', keep_default_na=False)
        assert len(integrada) == 3 and integrada.execucao_id.eq(execucao.name).all()
        for nome, rotulo in [('clinica', 'Clínica'), ('proteomica', 'Proteômica')]:
            origem = referencia.loc[referencia.modalidade.eq(nome)].iloc[0]
            destino = integrada.loc[integrada.modalidade.eq(rotulo)].iloc[0]
            for metrica in ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc']:
                for sufixo in ['media', 'dp']:
                    coluna = metrica + '_' + sufixo
                    np.testing.assert_allclose(float(destino[coluna]), float(origem[coluna]))
        texto = (execucao / 'relatorio_final_integrado.md').read_text(encoding='utf-8')
        links = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', texto)
        locais = [unquote(l) for l in links if not l.startswith(('https://', 'http://', '#'))]
        assert locais and all((execucao / l).is_file() for l in locais)
        evidencias['relatorio_integrado'] = dict(execucao_id=execucao.name, links_locais_conferidos=len(locais),
                                                metricas_conferidas=True)
    if inventario:
        entradas = json.loads(Path(inventario).read_text(encoding='utf-8'))
        for caminho, esperado in entradas.items():
            arquivo = Path(caminho)
            assert arquivo.stat().st_size == esperado['bytes'], caminho
            hash_atual = hashlib.sha256()
            with arquivo.open('rb') as origem:
                for bloco in iter(lambda: origem.read(8 * 1024 * 1024), b''):
                    hash_atual.update(bloco)
            assert hash_atual.hexdigest() == esperado['sha256'], caminho
        evidencias['entradas_preservadas_sha256'] = len(entradas)
    destino = execucao / 'verificacao_integridade.json'
    # Verificações repetidas preservam evidências anteriores.
    if destino.exists():
        raise FileExistsError(f'Verificação já registrada: {destino}')
    destino.write_text(json.dumps(evidencias, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(evidencias, ensure_ascii=False, indent=2))
    return evidencias


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execucao', required=True)
    parser.add_argument('--inventario', help='JSON com hashes das entradas originais')
    verificar(**vars(parser.parse_args()))
