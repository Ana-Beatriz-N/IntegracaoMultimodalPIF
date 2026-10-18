# Análise genômica e integração multimodal por Late Fusion

Esta etapa integra o TCC “Integração multimodal de dados genômicos e clínicos
via Machine Learning para predição de virulência da Peritonite Infecciosa
Felina (PIF)”. Seu objetivo é descrever sequências do vírus e preparar a
combinação das probabilidades de um modelo clínico e de um modelo genômico.
Late Fusion significa combinar as saídas dos modelos após sua avaliação
individual, em vez de juntar diretamente as variáveis das duas modalidades.

Este documento contém as instruções necessárias para instalar e executar a
etapa, inclusive obter os artefatos clínicos de entrada. Não requer a leitura
de outro README. Com os dados disponíveis, há auditoria genômica executada
e uma função de fusão preparada; não há classificador genômico treinado nem
Late Fusion validada.

A etapa multimodal não modifica o pipeline clínico, o notebook
`Pif_Dados_Clinicos_Reprodutivel.ipynb`, o CSV ou os resultados clínicos.
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
A modalidade clínica está implementada em `clinica_pipeline.py` e pode ser
executada também pelo notebook `Pif_Dados_Clinicos_Reprodutivel.ipynb`.

O alvo é **Healthy housemate (0) versus Caso incluído (1)**. Graus de suspeita
não confirmam PIF nem medem virulência. Experimentos ampliados com forma,
exames e tratamento podem revelar o processo de seleção e diagnóstico dos
grupos, em vez de informação disponível antes da suspeita de doença. Isso é
vazamento semântico e limita a interpretação de desempenhos elevados nesses
experimentos. A referência de integração é a regressão logística demográfica
pré-especificada, sem selecionar o modelo pelo maior resultado de CV.

Foi localizado somente `sequence.fasta`, com registro KX722530.1. Não foram
localizados metadados, alinhamentos ou correspondências clínicas documentadas.
O cabeçalho não permite atribuir animal, classe ou virulência. Por isso esta
entrega executa a descrição genômica, sem treinamento supervisionado artificial.

## Execução

Execute os comandos na raiz do projeto. É necessário Python com `venv` e
`pip`. As dependências compartilhadas estão em `requirements_clinicos.txt`:
NumPy, pandas, SciPy, scikit-learn, Matplotlib, joblib e ferramentas Jupyter.
Não há dependência adicional exclusiva desta etapa. Se já houver um ambiente
com essas dependências, pode reutilizá-lo. Para configurar um ambiente novo:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements_clinicos.txt
```

Entradas obrigatórias: um FASTA/FA (por padrão, `sequence.fasta`) e uma pasta
de resultados clínicos contendo `previsoes_oof.csv`, `comparacao_modelos.csv`
e `configuracao_versoes.json`. OOF (out-of-fold) significa que cada animal
recebe uma previsão de um modelo treinado sem o fold ao qual ele pertence.
Um fold é uma das partes reservadas para validação cruzada.

Se a pasta clínica usada na entrega estiver disponível:

```powershell
.\.venv\Scripts\python multimodal_pipeline.py --clinica resultados_clinicos/execucao_20261018T170043_012603Z
.\.venv\Scripts\python -m unittest verificar_multimodal -v
```

As pastas `resultados*/` são ignoradas pelo Git; em uma cópia nova do
repositório, os resultados locais podem não existir. Nesse caso, coloque na
raiz o CSV `06.2025_-_FIP_and_Housemate_Clinical_Data_Final_-_for_Dryad_Upload.csv`
e gere os artefatos executando:

```powershell
.\.venv\Scripts\python clinica_pipeline.py
```

Esse comando executa auditoria e avaliação clínica, criando uma nova pasta
`resultados_clinicos/execucao_.../`. Substitua o valor de `--clinica` pelo
caminho dessa pasta. Não precisa executar notebooks nem scripts auxiliares.
Não regenere resultados se já possuir artefatos válidos. A etapa multimodal
registra versões exatas, parâmetros e hashes em uma pasta nova.

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

`multimodal_pipeline.py` implementa a CLI de auditoria e a função
`late_fusion`; a CLI atual não treina modelos genômicos nem chama a fusão.
`verificar_multimodal.py` testa os cálculos e contratos com exemplos sintéticos.

Adicionados `multimodal_pipeline.py`, `verificar_multimodal.py` e este README.
Nenhum arquivo clínico foi alterado. Novas pastas em `resultados_multimodais`
contêm auditoria, features, cópia da referência clínica OOF, métricas clínicas
históricas, contagens de vínculos/duplicatas, tabelas de pareamento, relatório
e configuração com hashes. Métricas multimodais e previsões de fusão ficam
somente com cabeçalhos enquanto não houver dados válidos; não são zeros nem
resultados estimados. Dados de teste sintéticos ficam apenas no teste e nunca
entram no relatório científico.

Na execução `resultados_multimodais/execucao_20261018T214142_395402Z`, foi
auditado um genoma com 28.595 bases, GC de 38,67%, sem bases ambíguas, gaps
ou caracteres inválidos. Foram extraídas composição de nucleotídeos e
frequências dos 16 dinucleotídeos. Nenhum vínculo documental foi encontrado
entre esse genoma e os 72 animais clínicos: há zero pares, 72 registros
clínicos não pareados e um registro genômico não pareado. Nenhum registro
foi descartado. Esses números descrevem a execução disponível, não métricas
de predição multimodal.
