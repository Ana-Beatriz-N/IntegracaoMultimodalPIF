"""Testes de contratos com dados sintéticos, sem valor como resultado científico.

Execute: python -m unittest verificar_multimodal -v.
"""
import tempfile
from pathlib import Path
import unittest

import pandas as pd

from multimodal_pipeline import ALVO, auditar_genomas, late_fusion


class ContratosMultimodais(unittest.TestCase):
    """Verifica cálculos descritivos e barreiras contra integração inválida."""

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

    def test_fusao(self):
        """Une por ID mesmo em ordens diferentes e rejeita violações do contrato."""
        clinica = pd.DataFrame({"FIPID": ["a", "b", "c"], "alvo": [0, 1, 0],
            "fold": [1, 1, 2], "probabilidade": [.1, .8, .2], "particao_id": ["teste"]*3})
        genomica = clinica.iloc[[1, 0]].copy()
        genomica["probabilidade"] = [.6, .3]
        manifesto = {"alvo": ALVO, "origem": "oof", "grupos_conjuntos_auditados": True,
            "documentacao_particoes": "Exemplo sintético exclusivamente para teste", "particao_id": "teste"}
        pares, validas, metricas = late_fusion(clinica, genomica, manifesto)
        self.assertEqual(len(pares), 3)
        self.assertEqual(len(validas), 2)
        self.assertAlmostEqual(validas.set_index("FIPID").loc["a", "probabilidade_fusao"], .2)
        self.assertEqual(len(metricas), 3)
        for campo, valor in [("alvo", 0), ("fold", 9), ("probabilidade", 3), ("particao_id", "outra")]:
            invalida = genomica.copy()
            invalida.loc[invalida.FIPID.eq("b"), campo] = valor
            with self.subTest(campo=campo), self.assertRaises(ValueError):
                late_fusion(clinica, invalida, manifesto)
        with self.assertRaises(ValueError):
            late_fusion(clinica, pd.concat([genomica, genomica]), manifesto)
        with self.assertRaises(ValueError):
            late_fusion(clinica, genomica, {**manifesto, "origem": "ajuste_completo"})


if __name__ == "__main__":
    unittest.main()
