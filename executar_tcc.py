"""Executa as três modalidades independentes e registra sucesso, falhas e proveniência."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import traceback

from clinica.clinica_pipeline import Configuracao, ARQUIVO_PADRAO, executar as executar_clinica
from proteomica.proteomica_pipeline import ARQUIVO, executar as executar_proteomica
from genomica.genomica_pipeline import executar_integrado
from comparar_modalidades import comparar
from gerar_relatorio_integrado import gerar as gerar_relatorio, hash_arquivo


def executar(args):
    """Cria execução exclusiva; erros são registrados e as demais etapas continuam."""
    inicio = datetime.now(timezone.utc)
    pasta = Path(args.resultados) / 'execucoes' / inicio.strftime('execucao_%Y%m%dT%H%M%S_%fZ')
    pasta.mkdir(parents=True, exist_ok=False)
    raiz_codigo = Path(__file__).resolve().parent
    fontes = list(raiz_codigo.glob('*.py'))
    for modalidade in ['clinica', 'proteomica', 'genomica']:
        fontes.extend((raiz_codigo / modalidade).glob('*.py'))
    registro = dict(execucao_id=pasta.name, iniciada_em_utc=inicio.isoformat(), status='em_execucao',
        configuracao=vars(args), python=platform.python_version(),
        versoes={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'joblib']},
        codigo_sha256={p.relative_to(raiz_codigo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in fontes},
        etapas={})
    manifesto = pasta / 'manifesto_execucao.json'
    def salvar():
        manifesto.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding='utf-8')
    salvar()
    caminhos = {}
    etapas = {
        'clinica': lambda: executar_clinica(Configuracao(arquivo=args.clinica, resultados=str(pasta / 'clinicos')))['pasta'],
        'proteomica': lambda: executar_proteomica(args.proteomica, str(pasta / 'proteomicos')),
        'genomica': lambda: executar_integrado(args.genomica, args.metadados, pasta / 'genomicos', args.k),
    }
    for nome, funcao in etapas.items():
        print(f'Iniciando {nome}', flush=True)
        try:
            caminho = funcao()
            caminhos[nome] = caminho
            registro['etapas'][nome] = dict(status='concluida', pasta=caminho.relative_to(pasta).as_posix(),
                artefatos_sha256={p.relative_to(caminho).as_posix(): hash_arquivo(p)
                                  for p in caminho.rglob('*') if p.is_file()},
                escopo='exploratório sem classificador' if nome == 'genomica' else 'avaliação exploratória')
        except Exception as erro:
            registro['etapas'][nome] = dict(status='falhou', erro=str(erro))
            (pasta / f'erro_{nome}.txt').write_text(traceback.format_exc(), encoding='utf-8')
            print(f'Falha em {nome}: {erro}', flush=True)
        salvar()
    if len(caminhos) == 3:
        try:
            destino = comparar(caminhos['clinica'], caminhos['proteomica'], pasta / 'comparacao', caminhos['genomica'])
            registro['etapas']['comparacao'] = dict(status='concluida', pasta=destino.relative_to(pasta).as_posix())
        except Exception as erro:
            registro['etapas']['comparacao'] = dict(status='falhou', erro=str(erro))
            (pasta / 'erro_comparacao.txt').write_text(traceback.format_exc(), encoding='utf-8')
    else:
        registro['etapas']['comparacao'] = dict(status='nao_executada', motivo='Uma ou mais modalidades falharam; consultar logs')
    registro['status'] = 'concluida' if all(e['status'] == 'concluida' for e in registro['etapas'].values()) else 'incompleta'
    salvar()
    (pasta / 'requirements_executado.txt').write_text(''.join(f'{p}=={v}\n' for p, v in registro['versoes'].items()), encoding='utf-8')
    # Gera síntese também em execuções incompletas, sempre a partir deste manifesto.
    try:
        gerar_relatorio(pasta, args.resultados)
    except Exception as erro:
        registro['status'] = 'incompleta'
        registro['erro_relatorio_integrado'] = str(erro)
        (pasta / 'erro_relatorio_integrado.txt').write_text(traceback.format_exc(), encoding='utf-8')
        salvar()
    print(f'Execução {registro["status"]}: {pasta.resolve()}', flush=True)
    if registro['status'] != 'concluida':
        raise RuntimeError(f'Execução incompleta. Consulte {manifesto}')
    return pasta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clinica', default=ARQUIVO_PADRAO)
    parser.add_argument('--proteomica', default=ARQUIVO)
    parser.add_argument('--genomica', default='dados_genomicos')
    parser.add_argument('--metadados', default=None)
    parser.add_argument('--resultados', default='resultados')
    parser.add_argument('--k', type=int, choices=[1, 2, 3], default=2)
    executar(parser.parse_args())
