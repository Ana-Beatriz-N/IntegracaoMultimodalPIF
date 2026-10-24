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
import re
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gerar_metadados_genomicos import descobrir_fastas, ler_validar_fasta, ACCESSION


def sha256(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def ler_fasta(caminho):
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


def descrever_sequencia(seq, k):
    """Calcula qualidade e composição, mantendo denominadores explícitos."""
    c = Counter(seq)
    n = len(seq)
    canonicas = sum(c[b] for b in 'ACGT')
    ambiguas = sum(c[b] for b in 'RYSWKMBDHVN')
    janelas = Counter(seq[i:i+k] for i in range(max(0, n-k+1))
                      if set(seq[i:i+k]) <= set('ACGT'))
    validas = sum(janelas.values())
    qualidade = dict(comprimento=n, bases_canonicas=canonicas, bases_ambiguas=ambiguas,
        uracilas=c['U'], gaps=c['-']+c['.'],
        caracteres_invalidos=sum(v for b,v in c.items() if b not in 'ACGTURYSWKMBDHVN'),
        sequencia_vazia=n == 0, sha256_sequencia=hashlib.sha256(seq.encode()).hexdigest())
    atributos = dict(comprimento=n, fracao_ambigua=ambiguas/n if n else np.nan,
        fracao_uracila=c['U']/n if n else np.nan,
        gc=(c['G']+c['C'])/canonicas if canonicas else np.nan,
        janelas_validas=validas, fracao_janelas_validas=validas/(n-k+1) if n >= k else 0,
        **{f'freq_{b}': c[b]/canonicas if canonicas else np.nan for b in 'ACGT'},
        **{f'kmer_{"".join(p)}': janelas[''.join(p)]/validas if validas else np.nan
           for p in itertools.product('ACGT', repeat=k)})
    return qualidade, atributos


def auditar_genomas(caminhos, k=2):
    if k not in (1, 2, 3):
        raise ValueError('Use k entre 1 e 3.')
    auditoria, features = [], []
    for caminho in caminhos:
        for ordem, (cabecalho, seq) in enumerate(ler_fasta(caminho), 1):
            identidade = dict(arquivo=Path(caminho).as_posix(), registro=ordem,
                              sequence_id=cabecalho.split()[0])
            qualidade, atributos = descrever_sequencia(seq, k)
            auditoria.append(dict(**identidade, cabecalho=cabecalho, **qualidade))
            features.append(dict(**identidade, **atributos))
    if not auditoria:
        raise ValueError('Nenhuma sequência encontrada.')
    tabela = pd.DataFrame(auditoria)
    tabela['id_duplicado'] = tabela.sequence_id.duplicated(keep=False)
    tabela['sequencia_duplicada'] = tabela.sha256_sequencia.duplicated(keep=False)
    return tabela, pd.DataFrame(features)


def executar_auditoria_legada(args):
    auditoria, features = auditar_genomas(args.fasta, args.k)
    pasta = Path(args.saida) / datetime.now(timezone.utc).strftime("execucao_%Y%m%dT%H%M%S_%fZ")
    pasta.mkdir(parents=True, exist_ok=False)
    auditoria.to_csv(pasta / "auditoria_genomica.csv", index=False)
    features.to_csv(pasta / "features_genomicas.csv", index=False)
    contagens = {
        "registros_genomicos": len(auditoria),
        "registros_com_id_duplicado": int(auditoria.id_duplicado.sum()),
        "registros_com_sequencia_duplicada": int(auditoria.sequencia_duplicada.sum()),
        "descartados": 0,
        "status": "auditoria_descritiva_sem_classificador_genomico",
    }
    (pasta / "contagens.json").write_text(json.dumps(contagens, indent=2, ensure_ascii=False), encoding="utf-8")
    entradas = [*map(Path, args.fasta), Path(__file__).resolve()]
    config = {
        "k": args.k, "python": platform.python_version(),
        "versoes": {p: importlib.metadata.version(p) for p in ("numpy", "pandas")},
        "entradas_sha256": {str(p): sha256(p) for p in entradas},
    }
    (pasta / "configuracao_versoes.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (pasta / "RELATORIO.md").write_text(
        "# Auditoria genômica independente\n\n" + json.dumps(contagens, indent=2, ensure_ascii=False) +
        "\n\nNenhum classificador genômico foi treinado: faltam amostras independentes e rótulos diagnósticos documentados. "
        "O cabeçalho FASTA não determina animal, diagnóstico ou classe. Nenhuma sequência foi excluída.\n\n"
        f"Descritores: k={args.k}, {len(features.columns)-3} atributos incluindo comprimento e indicadores de qualidade. "
        "Não são evidência preditiva. Duplicatas exatas são auditadas; similaridade evolutiva, "
        "alinhamento e agrupamento por linhagem precisam de análise própria.\n\n"
        "Para comparar modelos genômicos: obter sequências por amostra independente, proveniência, "
        "rótulos e critérios diagnósticos, identificação de animal/coleta e grupos de dependência. "
        "Definir validação que respeite animais e similaridade genômica, com pré-processamento e seleção "
        "de atributos ajustados apenas no treino. Avaliar os mesmos classificadores e métricas usados nas "
        "demais bases, documentando as diferenças entre coortes e unidades amostrais.\n",
        encoding="utf-8")
    print(pasta.as_posix())


ESTADOS = {'confirmado', 'informado_nao_revisado', 'inferido_nome', 'desconhecido'}


def integrar_metadados(arquivos, pasta, metadados):
    tabela = pd.read_csv(metadados, sep=';', dtype=str, keep_default_na=False, encoding='utf-8-sig')
    obrigatorias = {'caminho_relativo', 'rotulo_inicial', 'validacao_diagnostico', 'referencia_diagnostico', 'cat_id'}
    if not obrigatorias <= set(tabela):
        raise ValueError(f'Metadados sem colunas necessárias: {obrigatorias - set(tabela)}')
    tabela['caminho_relativo'] = tabela.caminho_relativo.str.replace('\\', '/', regex=False)
    if tabela.caminho_relativo.duplicated().any():
        raise ValueError('Metadados com caminhos duplicados; revisar correspondência antes de executar.')
    existentes = {p.relative_to(pasta).as_posix() for p in arquivos}
    tabela['arquivo_encontrado'] = tabela.caminho_relativo.isin(existentes)
    # Linhas de metadados sem arquivo permanecem no relatório de integração.
    tabela = tabela.set_index('caminho_relativo', drop=False).astype(object)
    for caminho in sorted(existentes - set(tabela.index)):
        tabela.loc[caminho] = {c: '' for c in tabela.columns}
        tabela.loc[caminho, 'caminho_relativo'] = caminho
        tabela.loc[caminho, 'arquivo_encontrado'] = True
    for coluna in ['estado_diagnostico', 'rotulo_validado', 'estudo_id']:
        if coluna not in tabela:
            tabela[coluna] = ''
    for indice, linha in tabela.iterrows():
        estado = linha.estado_diagnostico.strip()
        if not estado:
            estado = 'inferido_nome' if linha.rotulo_inicial in {'0', '1'} else 'desconhecido'
        if estado not in ESTADOS:
            raise ValueError(f'Estado diagnóstico não reconhecido: {estado}, arquivo {indice}')
        tabela.loc[indice, 'estado_diagnostico'] = estado
        valido = (estado == 'confirmado' and linha.rotulo_validado in {'0', '1'}
                  and bool(linha.referencia_diagnostico.strip()))
        tabela.loc[indice, 'alvo_clinico_validado'] = linha.rotulo_validado if valido else ''
        tabela.loc[indice, 'diagnostico_utilizavel'] = valido
    return tabela


def executar_integrado(dados='dados_genomicos', metadados=None, saida='resultados_genomicos', k=2):
    if k not in (1, 2, 3):
        raise ValueError('Use k entre 1 e 3.')
    dados = Path(dados)
    metadados = Path(metadados) if metadados else dados / 'metadados_genomicos.csv'
    arquivos = descobrir_fastas(dados)
    if not arquivos:
        raise ValueError(f'Nenhum FASTA encontrado: {dados}')
    meta = integrar_metadados(arquivos, dados, metadados)
    auditorias, features, validacao = [], [], []
    for arquivo in arquivos:
        relativo = arquivo.relative_to(dados).as_posix()
        registros, problemas = ler_validar_fasta(arquivo)
        linha_meta = meta.loc[relativo]
        validacao.append(dict(caminho_relativo=relativo, quantidade_sequencias=len(registros),
                              observacoes='; '.join(problemas)))
        # O leitor tolerante permite registrar inclusive arquivos malformados.
        for ordem, (cabecalho, seq) in enumerate(registros, 1):
            qualidade, atributos = descrever_sequencia(seq, k)
            seq_id = f'{relativo}::registro_{ordem}'
            acesso = ACCESSION.search(cabecalho)
            # Divergência entre descrição e nome não altera o rótulo: requer revisão.
            descricao = cabecalho[1:].split(maxsplit=1)
            descricao = descricao[1] if len(descricao) > 1 else ''
            normalizar_nome = lambda s: re.sub('[^a-z0-9]', '', s.casefold())
            divergente = normalizar_nome(arquivo.stem) != normalizar_nome(descricao)
            auditorias.append(dict(amostra_id=seq_id, arquivo=relativo, registro=ordem,
                accession=acesso.group() if acesso else '', cabecalho=cabecalho,
                **qualidade,
                descricao_diverge_nome=divergente, estado_diagnostico=linha_meta.estado_diagnostico,
                rotulo_inicial=linha_meta.rotulo_inicial,
                alvo_clinico_validado=linha_meta.alvo_clinico_validado,
                cat_id=linha_meta.cat_id, estudo_id=linha_meta.estudo_id,
                grupo_potencial_nome=(re.search(r'fip_(\d+)_', arquivo.stem, re.I).group(1)
                    if re.search(r'fip_(\d+)_', arquivo.stem, re.I) else ''),
                qualidade_para_descricao=qualidade["comprimento"] > 0 and qualidade["caracteres_invalidos"] == 0,
                observacoes='; '.join(problemas)))
            features.append(dict(amostra_id=seq_id, **atributos))
    if not auditorias:
        raise ValueError('Nenhum registro FASTA; corrigir entradas antes de gerar descritores.')
    auditoria = pd.DataFrame(auditorias)
    auditoria['sequencia_duplicada'] = auditoria.sha256_sequencia.duplicated(keep=False)
    auditoria['accession_duplicado'] = auditoria.accession.ne('') & auditoria.accession.duplicated(keep=False)
    auditoria['cat_id_repetido'] = auditoria.cat_id.ne('') & auditoria.duplicated(['estudo_id', 'cat_id'], keep=False)
    def motivo(linha):
        causas = []
        if linha.alvo_clinico_validado == '':
            causas.append('Sem diagnóstico clínico confirmado e referenciado')
        if not linha.cat_id or not linha.estudo_id:
            causas.append('Unidade biológica/estudo não documentados')
        if not linha.qualidade_para_descricao:
            causas.append('Sequência vazia ou incompatível')
        if linha.sequencia_duplicada:
            causas.append('Sequência repetida requer adjudicação')
        return '; '.join(causas) or 'Requer planejamento de validação por animal e linhagem'
    auditoria['motivo_nao_treinamento'] = auditoria.apply(motivo, axis=1)
    atributos = pd.DataFrame(features)
    matriz = atributos.merge(auditoria[['amostra_id', 'arquivo', 'accession', 'cat_id', 'estudo_id',
        'alvo_clinico_validado', 'estado_diagnostico', 'motivo_nao_treinamento']], on='amostra_id', validate='one_to_one')
    pasta = Path(saida) / datetime.now(timezone.utc).strftime('execucao_%Y%m%dT%H%M%S_%fZ')
    pasta.mkdir(parents=True, exist_ok=False)
    for nome, tabela in [('auditoria_genomica', auditoria), ('features_genomicas', atributos),
                         ('matriz_genomica_exploratoria', matriz), ('metadados_integrados', meta),
                         ('validacao_arquivos', pd.DataFrame(validacao))]:
        tabela.to_csv(pasta / f'{nome}.csv', index=False, encoding='utf-8-sig')
    # O gráfico descreve sequências; não sugere separação diagnóstica confirmada.
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4))
    eixos[0].hist(auditoria.comprimento, bins=min(10, len(auditoria)))
    eixos[0].set(xlabel='Comprimento (nucleotídeos)', ylabel='Sequências')
    eixos[1].scatter(atributos.comprimento, atributos.gc)
    eixos[1].set(xlabel='Comprimento', ylabel='GC entre bases ACGT')
    fig.suptitle('Genômica exploratória: unidade = sequência, identidade animal pendente')
    fig.tight_layout()
    fig.savefig(pasta / 'qualidade_genomica.png', dpi=150)
    plt.close(fig)
    contagens = dict(arquivos=len(arquivos), sequencias=len(auditoria),
        arquivos_com_multiplas_sequencias=sum(v['quantidade_sequencias'] > 1 for v in validacao),
        arquivos_com_problemas=sum(bool(v['observacoes']) for v in validacao),
        rotulos_iniciais=meta.loc[meta.arquivo_encontrado.eq(True), 'rotulo_inicial'].value_counts().to_dict(),
        estados_diagnostico=meta.estado_diagnostico.value_counts().to_dict(),
        diagnosticos_confirmados=int(meta.diagnostico_utilizavel.eq(True).sum()),
        gatos_independentes_confirmados=None,
        sequencias_com_cat_id=int(auditoria.cat_id.ne('').sum()),
        sequencias_repetidas=int(auditoria.sequencia_duplicada.sum()),
        descricoes_divergentes=int(auditoria.descricao_diverge_nome.sum()),
        sequencias_nao_usadas_treinamento=len(auditoria),
        status='exploratorio_sem_classificador',
        motivo='Rótulos clínicos e independência biológica insuficientes; nenhuma métrica supervisionada calculada')
    (pasta / 'contagens.json').write_text(json.dumps(contagens, ensure_ascii=False, indent=2), encoding='utf-8')
    config = dict(k=k, unidade='sequência/amostra; não equivale a gato',
        atributos_preditores=list(atributos.columns.drop('amostra_id')),
        transformacao='Composição e k-mers ACGT normalizados por janelas válidas; sem posições não alinhadas',
        alinhadores={nome: shutil.which(nome) for nome in ['mafft', 'muscle', 'clustalo']},
        entradas_sha256={str(p): sha256(p) for p in [*arquivos, metadados, Path(__file__), Path('gerar_metadados_genomicos.py')]},
        python=platform.python_version(), versoes={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'matplotlib']})
    (pasta / 'configuracao_versoes.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    alertas = auditoria.loc[auditoria.descricao_diverge_nome | auditoria.sequencia_duplicada | auditoria.grupo_potencial_nome.ne(''),
                            ['arquivo', 'accession', 'descricao_diverge_nome', 'grupo_potencial_nome', 'sequencia_duplicada']]
    (pasta / 'RELATORIO.md').write_text(
        '# Genômica — análise exploratória integrada\n\n' + json.dumps(contagens, ensure_ascii=False, indent=2) +
        '\n\nA matriz tem uma linha por sequência; gatos independentes não estabelecidos. '
        'Rótulos iniciais não compõem o alvo validado. Colunas de identidade são rastreadores, nunca preditores. '
        'Não houve treinamento nem descarte silencioso de arquivos. Diagnósticos desconhecidos não viraram negativos. '
        'motivo_nao_treinamento documenta cada sequência. Arquivos sem sequência constam em validacao_arquivos.csv.\n\n'
        'Características: comprimento, GC e frequências ACGT/k-mers. Ambiguidades, U e caracteres inválidos '
        'não entram em janelas ACGT; frações/cobertura documentam a perda. A orientação foi preservada. '
        'Descritores globais não identificam variantes nem estabelecem homologia. Não foi aplicada seleção supervisionada. '
        'Sem referência validada, anotação de genes e alinhador, não há matriz de variantes posicionais. '
        'Nenhuma ferramenta externa é indispensável à análise implementada. Comprimento e cabeçalho não provam completude.\n\n'
        'Revisar origem, comparabilidade, orientação, estudo, diagnóstico e animal antes de treinar; '
        'amostras repetidas exigem grupos por gato, e linhagens/estudos exigem controle na validação.\n\n'
        'Alertas de descrição divergente, repetição e identificador potencial:\n\n' + alertas.to_csv(index=False), encoding='utf-8')
    print(f'Genômica exploratória: {len(arquivos)} arquivos, {len(auditoria)} sequências. Resultados: {pasta.resolve()}', flush=True)
    return pasta


def executar(args):
    if args.fasta:
        return executar_auditoria_legada(args)
    return executar_integrado(args.dados, args.metadados, args.saida, args.k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fasta', nargs='+', help='Auditoria descritiva avulsa sem integração clínica')
    parser.add_argument('--dados', default='dados_genomicos')
    parser.add_argument('--metadados')
    parser.add_argument('--saida', default='resultados_genomicos')
    parser.add_argument('--k', type=int, choices=[1, 2, 3], default=2)
    executar(parser.parse_args())
