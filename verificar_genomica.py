"""Testes de contratos com dados sintéticos, sem valor como resultado científico.

Execute: python -m unittest verificar_genomica -v.
"""
import tempfile
from pathlib import Path
import unittest

from genomica_pipeline import auditar_genomas, integrar_metadados, descrever_sequencia
from gerar_metadados_genomicos import classificar_nome, ler_validar_fasta, descobrir_fastas
import pandas as pd
import numpy as np
from avaliacao_comum import calcular_metricas


class AuditoriaGenomica(unittest.TestCase):
    """Verifica qualidade e cálculos descritivos das sequências."""

    def test_fasta(self):
        """Distingue ambiguidade, gaps, duplicatas e denominadores de k-mers."""
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / "teste.fa"
            arquivo.write_text(">a exemplo\nACGTNN-\n>b\nACGTNN-\n", encoding="utf-8")
            auditoria, features = auditar_genomas([arquivo])
        self.assertEqual(len(auditoria), 2)
        self.assertTrue(auditoria.sequencia_duplicada.all())
        self.assertEqual(auditoria.iloc[0].bases_ambiguas, 2)
        self.assertEqual(features.iloc[0].gc, .5)
        self.assertAlmostEqual(features.filter(like="kmer_").iloc[0].sum(), 1)
        self.assertEqual(features.iloc[0].janelas_validas, 3)

    def test_rotulos_ambiguos(self):
        for nome in ['non-FIP', 'no_FIP', 'FIPV', 'abcFIPxyz', 'FIP-negative']:
            self.assertEqual(classificar_nome(nome)[0], '')
        self.assertEqual(classificar_nome('isolate_fip_86')[0], 1)
        self.assertEqual(classificar_nome('controle')[0], 0)

    def test_integracao_real_nao_confirma_rotulos(self):
        dados = Path('dados_genomicos')
        arquivos = descobrir_fastas(dados)
        meta = integrar_metadados(arquivos, dados, dados / 'metadados_genomicos.csv')
        self.assertEqual(len(arquivos), 15)
        self.assertEqual(meta.rotulo_inicial.value_counts().to_dict(), {'1': 8, '0': 7})
        self.assertTrue(meta.alvo_clinico_validado.eq('').all())
        self.assertTrue(meta.estado_diagnostico.eq('inferido_nome').all())

    def test_metadado_ausente_e_confirmacao_explicita(self):
        with tempfile.TemporaryDirectory() as tmp:
            pasta = Path(tmp)
            arquivo = pasta / 'amostra.fa'
            arquivo.write_text('>AB123456.1\nACGT\n', encoding='utf-8')
            meta = pasta / 'meta.csv'
            colunas = ['caminho_relativo','rotulo_inicial','validacao_diagnostico',
                       'referencia_diagnostico','cat_id','estado_diagnostico','rotulo_validado']
            pd.DataFrame(columns=colunas).to_csv(meta, sep=';', index=False)
            tabela = integrar_metadados([arquivo], pasta, meta)
            self.assertEqual(tabela.iloc[0].estado_diagnostico, 'desconhecido')
            self.assertEqual(tabela.iloc[0].alvo_clinico_validado, '')
            pd.DataFrame([['amostra.fa','0','','','', 'confirmado','0']], columns=colunas).to_csv(meta, sep=';', index=False)
            tabela = integrar_metadados([arquivo], pasta, meta)
            self.assertEqual(tabela.iloc[0].alvo_clinico_validado, '')
            # Fonte fictícia apenas neste teste de contrato; nunca usada como dado científico.
            t = pd.read_csv(meta, sep=';', keep_default_na=False)
            t['referencia_diagnostico'] = 'referencia_de_teste'
            t.to_csv(meta, sep=';', index=False)
            self.assertEqual(integrar_metadados([arquivo], pasta, meta).iloc[0].alvo_clinico_validado, '0')

    def test_fasta_invalido_e_iupac(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'invalido.fa'
            p.write_text('ACGT\n>\nACGX\n>AB123456.1\n', encoding='utf-8')
            registros, problemas = ler_validar_fasta(p)
            self.assertEqual(len(registros), 2)
            self.assertGreaterEqual(len(problemas), 4)
        qualidade, _ = descrever_sequencia('ACGTURYSWKMBDHVN', 2)
        self.assertEqual(qualidade['caracteres_invalidos'], 0)

    def test_especificidade_e_auc(self):
        metricas = calcular_metricas([0,0,1,1], [0,1,1,1], [.1,.6,.8,.9])
        self.assertEqual(metricas['specificity'], .5)
        self.assertEqual(metricas['recall'], 1)
        self.assertEqual(metricas['roc_auc'], 1)
        self.assertTrue(np.isnan(calcular_metricas([1,1], [1,1], [.8,.9])['roc_auc']))


if __name__ == "__main__":
    unittest.main()
