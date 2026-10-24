"""Testes de falhas do relatório: exemplos temporários não são resultados científicos."""
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from gerar_relatorio_integrado import gerar, carregar_execucao


class RelatorioIntegrado(unittest.TestCase):
    def criar_manifesto(self, pasta, etapas):
        """Registra uma execução de teste sem fabricar tabelas científicas."""
        pasta.mkdir()
        m = dict(execucao_id=pasta.name, iniciada_em_utc='data_de_teste', status='incompleta',
                 configuracao={}, etapas=etapas)
        (pasta / 'manifesto_execucao.json').write_text(json.dumps(m), encoding='utf-8')

    def test_falhas_nao_reutilizam_relatorio_antigo(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            execucao = base / 'execucao_teste'
            self.criar_manifesto(execucao, {'clinica': {'status': 'falhou', 'erro': 'entrada ausente'}})
            saida = base / 'resultados'
            saida.mkdir()
            (saida / 'relatorio_final_integrado.md').write_text('RELATORIO_ANTIGO', encoding='utf-8')
            caminho = gerar(execucao, saida)
            texto = caminho.read_text(encoding='utf-8')
            self.assertNotIn('RELATORIO_ANTIGO', texto)
            self.assertIn('entrada ausente', texto)
            tabela = pd.read_csv(saida / 'tabela_comparativa.csv', keep_default_na=False)
            self.assertEqual(len(tabela), 3)
            self.assertTrue(tabela.f1.eq('N/A').all())
            antes = caminho.read_bytes()
            gerar(resultados=saida)
            self.assertEqual(antes, caminho.read_bytes())

    def test_rejeita_etapa_de_outra_execucao(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            execucao = base / 'execucao_teste'
            externa = base / 'outra_execucao'
            externa.mkdir()
            self.criar_manifesto(execucao, {'clinica': {'status': 'concluida', 'pasta': str(externa)}})
            _, dados = carregar_execucao(execucao)
            self.assertIsNone(dados['clinica']['pasta'])
            self.assertIn('mistura', dados['clinica']['motivo'])

    def test_artefato_ausente_vira_parcial(self):
        with tempfile.TemporaryDirectory() as tmp:
            execucao = Path(tmp) / 'execucao_teste'
            self.criar_manifesto(execucao, {'genomica': {'status': 'concluida', 'pasta': 'genomicos',
                'artefatos_sha256': {'contagens.json': 'hash_de_teste'}}})
            _, dados = carregar_execucao(execucao)
            self.assertTrue(dados['genomica']['status'].startswith('Parcial'))
            self.assertIsNone(dados['genomica']['pasta'])


if __name__ == '__main__':
    unittest.main()
