"""Gera metadados exploratórios sem inferir diagnóstico clínico confirmado.

Uso: python -m genomica.gerar_metadados_genomicos --dados dados_genomicos
CSV: UTF-8 com BOM e separador ponto e vírgula para Excel em português.
Saídas existentes nunca são sobrescritas; escolha outra pasta com --saida.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import re

EXTENSOES = {'.fasta', '.fa', '.fna', '.fas'}
IUPAC = set('ACGTURYSWKMBDHVN')
MARCADOR = re.compile(r'(?<![A-Za-z0-9])FIP(?![A-Za-z0-9])', re.I)
AMBIGUO = re.compile(
    r'(?<![A-Za-z0-9])(?:non|no|not|sem)[\s_.-]*FIP(?![A-Za-z0-9])'
    r'|(?<![A-Za-z0-9])(?:FIPV|FECV)(?![A-Za-z0-9])'
    r'|(?<![A-Za-z0-9])FIP[\s_.-]*(?:negative|negativo|neg)(?![A-Za-z0-9])', re.I)
ACCESSION = re.compile(r'(?<![A-Za-z0-9_])(?:[A-Z]{1,2}_\d{5,9}|[A-Z]\d{5}|[A-Z]{2}\d{6,8})(?:\.\d+)?(?![A-Za-z0-9_])')
CAMPOS = [
    'arquivo_fasta', 'caminho_relativo', 'accession', 'cabecalho_fasta',
    'quantidade_sequencias', 'comprimento_sequencia', 'comprimentos_sequencias',
    'rotulo_inicial', 'diagnostico_inicial', 'fonte_rotulo',
    'validacao_diagnostico', 'referencia_diagnostico', 'cat_id', 'observacoes',
]


def descobrir_fastas(pasta):
    """Busca recursivamente, inclusive extensões em maiúsculas."""
    return sorted((p for p in pasta.rglob('*') if p.is_file()
                   and p.suffix.lower() in EXTENSOES), key=lambda p: p.as_posix().lower())


def classificar_nome(nome):
    """Deixa rótulo vazio em convenções ambíguas, exigindo revisão humana."""
    alertas = []
    if AMBIGUO.search(nome) or (re.search('fip', nome, re.I) and not MARCADOR.search(nome)):
        return '', 'Indeterminado (revisar nome)', ['Nome ambíguo: rótulo inicial não atribuído']
    positivo = bool(MARCADOR.search(nome))
    if not positivo:
        alertas.append('Ausência de FIP no nome não confirma ausência clínica de PIF')
    return int(positivo), 'PIF' if positivo else 'Sem PIF (provisório)', alertas


def ler_validar_fasta(caminho):
    """Preserva arquivos originais; registra problemas sem descartar registros."""
    registros, alertas = [], []
    try:
        conteudo = caminho.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        conteudo = caminho.read_bytes().decode('utf-8-sig', errors='replace')
        alertas.append('Codificação UTF-8 inválida; caracteres ilegíveis substituídos apenas na leitura')
    except OSError as erro:
        return [], [f'Falha de leitura: {erro}']
    if not conteudo.strip():
        return [], ['Arquivo vazio']
    cabecalho, partes = None, []
    for numero, linha in enumerate(conteudo.splitlines(), 1):
        linha = linha.strip()
        if not linha:
            continue
        if linha.startswith('>'):
            if cabecalho is not None:
                registros.append((cabecalho, ''.join(partes)))
            cabecalho, partes = linha, []
            if not linha[1:].strip():
                alertas.append(f'Cabeçalho vazio na linha {numero}')
        elif cabecalho is None:
            alertas.append(f'Conteúdo sem cabeçalho FASTA na linha {numero}')
        else:
            partes.append(''.join(linha.split()).upper())
    if cabecalho is not None:
        registros.append((cabecalho, ''.join(partes)))
    if not registros:
        alertas.append('Nenhum cabeçalho iniciado por > encontrado')
    for indice, (cab, seq) in enumerate(registros, 1):
        if not seq:
            alertas.append(f'Sequência vazia no registro {indice}')
        invalidos = Counter(c for c in seq if c not in IUPAC)
        if invalidos:
            alertas.append(f'Caracteres incompatíveis no registro {indice}: {dict(sorted(invalidos.items()))}')
        if not ACCESSION.search(cab):
            alertas.append(f'Cabeçalho sem accession NCBI reconhecível no registro {indice}')
    if len(registros) > 1:
        alertas.append('Múltiplas sequências: comprimentos na ordem dos registros; accession refere-se ao primeiro')
    return registros, alertas


def analisar_arquivo(caminho, pasta):
    registros, problemas = ler_validar_fasta(caminho)
    rotulo, diagnostico, alertas = classificar_nome(caminho.stem)
    cabecalho = registros[0][0] if registros else ''
    acesso = ACCESSION.search(cabecalho)
    # Nomes de isolados não comprovam identidade do gato nem independência amostral.
    if re.search(r'(?<![a-z0-9])fip_\d+_(?:lung|liver|mln)(?![a-z0-9])', caminho.stem, re.I):
        alertas.append('Nome sugere identificador e tecido; confirmar gato e possível repetição entre arquivos')
    linha = dict(zip(CAMPOS, [''] * len(CAMPOS)))
    linha.update(
        arquivo_fasta=caminho.name, caminho_relativo=caminho.relative_to(pasta).as_posix(),
        accession=acesso.group() if acesso else '', cabecalho_fasta=cabecalho,
        quantidade_sequencias=len(registros),
        comprimento_sequencia=len(registros[0][1]) if len(registros) == 1 else '',
        comprimentos_sequencias='|'.join(str(len(s)) for _, s in registros) if len(registros) > 1 else '',
        rotulo_inicial=rotulo, diagnostico_inicial=diagnostico, fonte_rotulo='Nome do arquivo',
        validacao_diagnostico='Pendente de confirmação',
        observacoes='; '.join(problemas + alertas),
    )
    return linha, problemas


def gerar(pasta, saida):
    """Gera uma linha por arquivo e relatório; recusa sobrescrita."""
    if not pasta.is_dir():
        raise ValueError(f'Diretório de dados inexistente: {pasta}')
    arquivos = descobrir_fastas(pasta)
    if not arquivos:
        raise ValueError(f'Nenhum FASTA encontrado em {pasta}')
    destinos = [saida / 'metadados_genomicos.csv', saida / 'relatorio_validacao_genomica.txt']
    existentes = [str(p) for p in destinos if p.exists()]
    if existentes:
        raise FileExistsError('Saídas existentes; nada foi sobrescrito. Use --saida: ' + ', '.join(existentes))
    linhas, problemas = [], []
    for arquivo in arquivos:
        linha, erros = analisar_arquivo(arquivo, pasta)
        linhas.append(linha)
        if erros:
            problemas.append(linha['caminho_relativo'])
    contagens = Counter(linha['rotulo_inicial'] for linha in linhas)
    multiplos = [l['caminho_relativo'] for l in linhas if l['quantidade_sequencias'] > 1]
    ambiguos = [l['caminho_relativo'] for l in linhas if l['rotulo_inicial'] == '']
    relatorio = [
        'VALIDAÇÃO GENÔMICA — FCoV', f'Arquivos processados: {len(linhas)}',
        f'Rótulo inicial 1 (possível PIF): {contagens[1]}',
        f'Rótulo inicial 0 (possível controle): {contagens[0]}',
        f'Rótulo não atribuído (nome ambíguo): {len(ambiguos)}',
        f'Arquivos com problemas de FASTA: {len(problemas)}',
        f'Arquivos com múltiplas sequências: {len(multiplos)}',
        f'Nomes ambíguos: {len(ambiguos)}',
        '', 'Problemas de FASTA:', *(problemas or ['Nenhum']),
        '', 'Múltiplas sequências:', *(multiplos or ['Nenhum']),
        '', 'Nomes ambíguos:', *(ambiguos or ['Nenhum']),
        '', 'Revisão manual: TODOS os arquivos requerem confirmação clínica do rótulo.',
        'PIF no nome é apenas hipótese inicial; ausência de FIP não comprova controle saudável.',
        'cat_id vazio: nomes de isolados não foram assumidos como identificadores de gatos.',
        'Accessions são extraídos por formato, sem consulta ou validação externa.',
        'Comprimentos não comprovam que um genoma seja completo. Nenhum arquivo foi descartado.',
        '', 'Observações por arquivo:',
    ]
    relatorio.extend(f'{l["caminho_relativo"]}: {l["observacoes"] or "Sem alertas adicionais"}' for l in linhas)
    saida.mkdir(parents=True, exist_ok=True)
    with destinos[0].open('x', encoding='utf-8-sig', newline='') as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=CAMPOS, delimiter=';')
        escritor.writeheader()
        escritor.writerows(linhas)
    with destinos[1].open('x', encoding='utf-8', newline='') as arquivo:
        arquivo.write('\n'.join(relatorio) + '\n')
    print('\n'.join(relatorio[:9]))
    for destino in destinos:
        print(destino.resolve())
    return linhas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dados', type=Path, required=True, help='Pasta dos FASTA; busca recursiva')
    parser.add_argument('--saida', type=Path, help='Pasta de saída; padrão: pasta dos FASTA')
    args = parser.parse_args()
    try:
        gerar(args.dados, args.saida if args.saida is not None else args.dados)
    except (OSError, ValueError) as erro:
        parser.exit(1, f'Erro: {erro}\n')


if __name__ == '__main__':
    main()
