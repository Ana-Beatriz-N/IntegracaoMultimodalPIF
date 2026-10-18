# Continuação genômica e Late Fusion

O pipeline clínico, notebooks, CSV e resultados anteriores permanecem intactos.
Não é necessário reexecutar a clínica nem carregar os modelos joblib.

## Auditoria prévia e ponto de integração

`clinica_pipeline.py` usa FIPID como identificador e Home number ID como grupo
de validação. `avaliar_modelos` exporta alvo, fold, previsão, escore e tipo de
escore em `previsoes_oof.csv`. Há Dummy, regressão logística, Random Forest e
SVM nos experimentos demográfico e ampliados. A SVM fornece margem, não
probabilidade. `ComponenteClinico.prever` fornece probabilidades para novos
dados; o componente serializado foi ajustado na coorte completa e não deve
produzir escores de teste nessa própria coorte.

Foram encontradas duas execuções clínicas, terminadas em `165656_992685Z` e
`170043_012603Z`, ambas em 18/10/2026. A execução usada abaixo é explícita,
sem escolha automática por desempenho. A coorte descrita tem 72 animais,
43 casas, 43 controles e 29 casos incluídos; validação em cinco folds por casa.
O notebook clínico original contém trechos de proteômica; a implementação
reproduzível e seu notebook separado já resolvem a modalidade clínica.

O alvo é **Healthy housemate (0) versus Caso incluído (1)**. Graus de suspeita
não confirmam PIF nem medem virulência. Experimentos ampliados com forma,
exames e tratamento têm limitações de vazamento semântico já documentadas
na clínica. A referência de integração é a regressão logística demográfica
pré-especificada, sem selecionar o modelo pelo maior resultado de CV.

Foi localizado somente `sequence.fasta`, com registro KX722530.1. Não foram
localizados metadados, alinhamentos ou correspondências clínicas documentadas.
O cabeçalho não permite atribuir animal, classe ou virulência. Por isso esta
entrega executa a descrição genômica, sem treinamento supervisionado artificial.

## Execução

Use o ambiente clínico existente; as dependências estão em
`requirements_clinicos.txt`. Não há nova dependência. Cada execução registra
versões exatas, parâmetros e hashes em uma pasta nova.

```powershell
python multimodal_pipeline.py --clinica resultados_clinicos/execucao_20261018T170043_012603Z
python -m unittest verificar_multimodal -v
```

Para outros FASTA: `--fasta dados/a.fa dados/b.fasta`; `--k 2` é o padrão
(16 frequências de dinucleotídeos); `--saida resultados_multimodais` controla
o diretório. As entradas são somente lidas. Composição e GC usam somente
ACGT; k-mers usam somente janelas canônicas válidas, não atravessam N/gaps.
As frequências não resolvem homologia, cobertura desigual, recombinação ou
efeito de linhagem. Não comparar diretamente genomas completos e fragmentos
sem definir uma região comparável. Duplicatas são registradas, sem exclusão.

## Contrato da fusão preparada

`late_fusion(clinica, genomica, manifesto)` recebe DataFrames com `FIPID`,
`alvo`, `fold`, `probabilidade`, `particao_id`. Não associa linhas por posição.
O identificador genômico precisa antes ser traduzido por tabela documental
`sequence_id,FIPID,fonte_correspondencia`, com animal, coleta, casa e procedência.
Essa tabela não existe nos dados atuais e não é fabricada pela execução.
IDs repetidos interrompem a fusão para revisão, sem agregar recoletas.

O manifesto exige `alvo` igual a `ALVO`, `origem="oof"`,
`grupos_conjuntos_auditados=True`, `documentacao_particoes` e `particao_id`.
Este último identifica a divisão completa de treino/validação, incluindo
grupos e coorte; coincidir somente o número de fold é insuficiente. A função
valida o contrato declarado, mas não consegue provar a procedência dos escores:
é responsabilidade do produtor documentar IDs de treino/teste e agrupamentos.
Não renomear escores de ajuste como OOF. Não transformar margem SVM em
probabilidade por recorte ou sigmoid arbitrária.

Após auditar vínculos e semelhança genômica, treinar baseline Dummy e modelos
genômicos fixos como regressão logística e Random Forest dentro de folds
comuns, com escala/seleção aprendidas apenas no treino. É necessário ter duas
classes e grupos independentes suficientes em treino e validação; a simples
existência de duas sequências não basta. Se parentesco viral cruzar os folds
clínicos existentes, será necessário recalcular a avaliação clínica com grupos
comuns usando suas funções existentes, documentando essa alteração mínima.
Não há motivo para implementar ou executar esse treinamento com um genoma
sem rótulo. A automação de treinamento futuro depende dessa auditoria.

A fusão usa média simples (peso 0,5 por modalidade), pré-fixada para evitar
ajuste de peso na avaliação. Limiar também fixado em 0,5. Calibração, pesos ou
limiares alternativos exigem validação interna ao treino. A função retorna
auditoria externa de IDs, previsões pareadas e métricas por fold nas mesmas
amostras para as três modalidades. Resumir dispersão entre folds e não
tratá-la como intervalo de confiança. Não alegar ganho com base em uma
seleção posterior de modelos no mesmo OOF.

## Artefatos e mudanças

Adicionados `multimodal_pipeline.py`, `verificar_multimodal.py` e este README.
Nenhum arquivo clínico foi alterado. Novas pastas em `resultados_multimodais`
contêm auditoria, features, cópia da referência clínica OOF, métricas clínicas
históricas, contagens de vínculos/duplicatas, tabelas de pareamento, relatório
e configuração com hashes. Métricas multimodais e previsões de fusão ficam
somente com cabeçalhos enquanto não houver dados válidos; não são zeros nem
resultados estimados. Dados de teste sintéticos ficam apenas no teste e nunca
entram no relatório científico.
