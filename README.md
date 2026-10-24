# Comparação de modelos de Machine Learning baseados em dados clínicos, proteômicos e genômicos para a classificação da Peritonite Infecciosa Felina (PIF)

## Título e objetivo do TCC

Avaliar modelos de classificação utilizando dados clínicos, proteômicos e genômicos do coronavírus felino (FCoV), documentando os limites científicos de cada base. Este é o **único README do projeto**. A documentação clínica, proteômica e genômica está consolidada neste arquivo.

## Descrição geral do projeto

As três modalidades são analisadas separadamente, em conjuntos sem pareamento confiável. A clínica permite avaliação exploratória de grupos de suspeita; a proteômica permite demonstração em pools; a genômica permite auditoria e descritores de sequência. Isso não constitui validação de diagnóstico individual de PIF confirmada. Não há late fusion nem comparação controlada nos mesmos gatos.

## Objetivos da análise

- Auditar entradas, rótulos, exclusões, ausências, unidades e dependências entre registros.
- Preservar algoritmos e critérios metodológicos, com transformações aprendidas somente no treino.
- Calcular métricas, previsões OOF, matrizes e gráficos onde os dados permitem avaliação supervisionada.
- Integrar resultados reais de uma única execução em um relatório principal e uma tabela comparativa.
- Expor falhas, resultados parciais e métricas não calculáveis, sem inventar dados ou reutilizar resultados antigos.

## Estrutura de diretórios e arquivos

```text
README.md                         documentação principal
executar_tcc.py                    execução completa e manifesto
gerar_relatorio_integrado.py       síntese central reutilizável
clinica_pipeline.py                análise e componente clínico; modelos compartilhados
proteomica_pipeline.py             matriz quantitativa, seleção e avaliação de pools
genomica_pipeline.py               auditoria FASTA, descritores e integração de metadados
gerar_metadados_genomicos.py        geração inicial de metadados, sem sobrescrita
avaliacao_comum.py                 métricas binárias compartilhadas
comparar_modalidades.py            tabelas técnicas de comparação e referência fixa
verificar_*.py                     verificações e testes de integridade/falhas
requirements_clinicos.txt          dependências de todas as análises
Pif_Dados_Clinicos_Reprodutivel.ipynb  interface opcional para o módulo clínico
06.2025_-_FIP_and_Housemate_Clinical_Data_Final_-_for_Dryad_Upload.csv
JPST003448_all/                    TSV quantitativo e originais MGF/mzXML
dados_genomicos/                   FASTA e metadados originais
resultados/
  execucao_atual.json              apontador explícito da execução publicada
  relatorio_final_integrado.md     relatório principal da execução indicada
  tabela_comparativa.csv           três linhas, mesmas fontes do relatório
  execucoes/execucao_DATA/
    manifesto_execucao.json        origem, status, caminhos e hashes dos artefatos
    requirements_executado.txt     versões efetivas
    clinicos/execucao_DATA/        artefatos e relatório técnico clínico
    proteomicos/execucao_DATA/     artefatos e relatório técnico proteômico
    genomicos/execucao_DATA/       artefatos e relatório técnico genômico
    comparacao/execucao_DATA/      comparação técnica de todos os modelos
    relatorio_final_integrado.md   cópia preservada desta execução
    tabela_comparativa.csv         cópia preservada desta execução
```

Somente os resultados da execução `execucao_20261024T173849_125977Z` foram mantidos nesta limpeza, em `resultados/execucoes/`, junto ao relatório integrado, à tabela e ao apontador atual em `resultados/`. As CLIs avulsas ainda aceitam destinos configuráveis e podem criar novas pastas. As pastas de resultados são ignoradas pelo Git: uma cópia nova precisa executar o pipeline ou receber os artefatos completos.

Os READMEs secundários, os documentos Markdown de auditoria e o relatório da execução anterior foram removidos a pedido. METODOLOGIA.md aponta para este documento. Os inventários de entradas e da limpeza anterior foram preservados como registros auxiliares. O README de `node_modules/pandas` pertence a uma dependência JavaScript e não integra a documentação científica do projeto.

## Descrição das fontes de dados

|Modalidade|Entrada real|Unidade disponível|Proveniência documentada|
|---|---|---|---|
|Clínica|CSV FIP and Housemate Clinical Data na raiz|Registro de gato, com casa disponível|O nome menciona Dryad; DOI/artigo e dicionário completo não estão documentados localmente|
|Proteômica|`JPST003448_all/21905517_merge-Metaboanalyst.txt`|Pool/subgrupo|Depósito JPST003448 e artigo relacionado na documentação histórica|
|Genômica|FASTA em `dados_genomicos/` e `metadados_genomicos.csv`|Sequência/amostra; gato independente não estabelecido|Accessions extraídos dos cabeçalhos; estudos e diagnósticos precisam de adjudicação|

Não são baixados dados ou consultadas APIs durante a execução. Accessions reconhecidos por formato não são validação de origem, completude ou diagnóstico. Consulte os cabeçalhos e metadados derivados para rastrear cada sequência.

## Modalidade clínica

A entrada disponível tem 100 registros e 12 colunas. Entram somente Include e Healthy housemate: 72 registros elegíveis, 43 controles informados e 29 suspeitos; 28 exclusões originais são respeitadas. Os positivos incluem 16 Very likely, 7 Highly suspicious e 6 Suspicious. Essas categorias não são convertidas em PIF confirmada. NA na categoria diagnóstica dos controles não comprova ausência de doença.

FIPID identifica o registro; Home number ID agrupa por casa. IDs ausentes/repetidos, casa ausente, categorias de inclusão ou combinações diagnóstico/grupo inesperadas interrompem a análise. FIPID único não garante que IDs diferentes não sejam recoletas do mesmo gato. Somente a casa está disponível como agrupamento confiável operacional.

Referência: idade, sexo e raça. Experimentos ampliados acrescentam forma, resultado/realização de exames e tratamento, separados por risco de vazamento semântico. Podem revelar seleção/investigação em vez de previsão prospectiva. Não são usados para escolher a referência comparativa.

## Modalidade proteômica

Usa o TSV quantitativo do depósito, não MGF/mzXML diretamente. A linha Label contém 11 FIP e 6 Healthy. A matriz tem 17 pools nas linhas e 3384 atributos anotados pelo depositante nas colunas. Identificadores/anotações não são abundâncias preditoras. As 606 linhas totalmente vazias são removidas; registros com ID ausente/duplicado exigem revisão, sem agregação silenciosa.

O artigo associado, conforme documentação histórica, descreve agrupamento/seleção por perfil espectral anterior à tabela. Os 17 pools não equivalem a 17 gatos. Sem composição dos pools não há agrupamento por animal confiável, e a CV não desfaz a seleção anterior. A identificação proteica, inferência e FDR não são repetidos pelo projeto.

Os 379 MGF e 379 mzXML são inventariados estruturalmente. Alertas sobre ausência de espectros/scans não removem automaticamente atributos quantitativos porque falta mapa validado de corridas para pools. A seleção de atributos por ANOVA é computacional e não identifica biomarcadores validados.

## Modalidade genômica

Busca recursivamente `.fasta`, `.fa`, `.fna` e `.fas`, sem alterar originais. A base atual tem 15 arquivos com uma sequência cada. O nome fornece oito hipóteses positivas e sete possíveis controles; não há confirmação clínica, estudo ou cat_id documentados. Nomes FIP, accessions e sequências diferentes não comprovam diagnósticos ou gatos independentes. Convenções ambíguas como non-FIP, no_FIP e FIPV ficam sem rótulo inicial no gerador.

Estados explícitos: `confirmado`, `informado_nao_revisado`, `inferido_nome` e `desconhecido`. Campos opcionais para adjudicação: `estado_diagnostico`, `rotulo_validado` (0/1), `estudo_id`, `cat_id` e `referencia_diagnostico`. O alvo clínico só é preenchido com estado confirmado, rótulo válido e referência preenchida; essa verificação estrutural não valida a veracidade clínica da fonte. O rótulo inicial nunca é promovido automaticamente a alvo confirmado.

Arquivos sem metadados ficam desconhecidos; metadados sem arquivo permanecem na integração. Arquivos vazios/malformados, sequências vazias, caracteres inválidos, múltiplos registros, duplicatas, accession repetido, divergência de descrição e nomes que sugerem tecidos do mesmo gato são documentados. Grupos sugeridos pelo nome não são cat_id confirmado. Todos os registros permanecem rastreáveis, com motivo de não treinamento.

Nos dados atuais, não há classificador genômico treinado: somente descritores, qualidade e integração. Preencher campos não ativa treinamento automaticamente; uma futura avaliação requer planejamento por animal, estudo e linhagem. Não há matriz de variantes, anotação por gene ou comparação posicional sem alinhamento validado.

## Pré-processamento de cada modalidade

**Clínica:** leitura como texto preserva NA e grafias, com normalização de espaços/caixa. Idade por nascimento/coleta em mês/dia/ano; datas inválidas/futuras e idades fora de 0–30 anos ficam ausentes, com auditoria. No treino: mediana, StandardScaler e OneHotEncoder com ausência explícita e categorias inéditas ignoradas. Coluna numérica totalmente ausente no treino usa o tratamento operacional de `keep_empty_features`, não uma idade real. Sexo mantém MN/FS/MI/FI; raças não são fundidas sem dicionário. Exames viram positivo, negativo, não realizado ou ausente; títulos/datas/material não são preditores. NA de forma ou tratamento não significa ausência clínica. IDs, casa, grupo e diagnóstico não entram como características.

**Proteômica:** escala fornecida preservada, sem novo log ou normalização global não documentada. Duas hipóteses fixas: zeros mantidos e zeros como ausentes. Dentro do treino: observações em pelo menos 50% das amostras, imputação por mediana, remoção de constantes, ANOVA até 10 atributos e StandardScaler. Não se escolhe a hipótese de zeros pelo maior desempenho.

**Genômica:** códigos IUPAC válidos aceitos; U não é convertido silenciosamente em T. Comprimento, GC, composição ACGT, frações ambígua/uracila, cobertura e frequências de k-mers, padrão k=2. Somente janelas ACGT entram no denominador; cobertura exportada torna a perda explícita. Orientação original preservada. São 26 descritores no padrão atual, sem usar nome, accession, estudo ou rótulo inicial como preditores. Não se equiparam posições não alinhadas; k-mers não são variantes homólogas. MAFFT/MUSCLE/Clustal Omega são procurados e registrados, mas não instalados ou indispensáveis aos descritores globais. Completude e regiões comparáveis precisam de confirmação na fonte.

## Algoritmos de Machine Learning utilizados

|Modelo|Parâmetros mantidos|
|---|---|
|Dummy|strategy=prior|
|Regressão logística|C=1, max_iter=2000, class_weight=balanced|
|SVM linear|C=1, kernel=linear, class_weight=balanced|
|Random Forest|200 árvores, profundidade 3, folha mínima 4, class_weight=balanced|

Semente 42 quando aplicável. Sem algoritmos adicionais, SMOTE, calibração ou busca de hiperparâmetros. A genômica atual não avalia classificadores supervisionados.

## Estratégia de treinamento e validação

Clínica: StratifiedGroupKFold determinístico por casa, até cinco folds, exigindo ambas as classes em todo treino/validação; número reduzido se necessário. Proteômica: três folds estratificados de pools, shuffle=True e semente 42, com as mesmas partições para todos os modelos e hipóteses. Imputação, escala, codificação e seleção são ajustadas apenas no treino. OOF fornece uma previsão por unidade/modelo/experimento.

Associações clínicas usam permutações dentro de casa, hipótese de permutabilidade condicional, correção +1 e Benjamini–Hochberg na família exportada. Casas sem contraste não informam o teste. V de Cramér e diferenças médias são descritivos; ausências podem representar coleta. A correção BH não garante controle sob qualquer dependência. Importância por permutação usa validação, agrega repetições dentro do fold e não orienta seleção ou ajustes; perturbações rompem correlações entre variáveis/animais. Coeficientes OneHot são exportados por categoria/fold, sem interpretação causal.

O componente logístico demográfico final é ajustado na coorte inteira para reutilização, sem avaliação independente desse ajuste. Carregue joblib apenas de origem confiável. `ComponenteClinico.prever` preserva FIPID e aceita categorias inéditas; não exige grupo ou diagnóstico, mas precisa dos campos brutos usados pela engenharia. O carregador de pesquisa valida o esquema completo. Não há modelo final proteômico ou genômico para uso diagnóstico.

## Métricas de avaliação

Accuracy, precisão da classe positiva, sensibilidade/recall, especificidade TN/(TN+FP), F1 e ROC-AUC, por fold. Médias não ponderadas e desvios amostrais não são intervalos de confiança. Precisão/F1 sem positivos previstos recebem zero explicitamente. Especificidade sem negativos e AUC sem duas classes ficam indisponíveis. AUC da SVM usa margem em cada fold, nunca margens agregadas de modelos distintos. Matrizes OOF usam linhas observadas, colunas previstas e ordem 0/1. Não há teste externo.

No relatório integrado, **N/A** significa não calculado ou não aplicável; o motivo está no status e no texto de cada modalidade. Falha nunca vira zero, e desconhecido nunca vira negativo. CSVs técnicos antigos podem usar “Indisponível” com o mesmo significado.

## Comparação entre modalidades

O relatório integrado inclui uma tabela de três linhas e tabelas completas por modelo/experimento. A referência pré-especificada permanece regressão logística clínica demográfica e proteômica com zeros mantidos; genômica fica N/A. Os números vêm de `comparacao_modelos.csv`, conferidos contra `metricas_folds.csv`, sem valores de desempenho escritos manualmente no gerador.

O maior F1 observado dentro do experimento é mostrado como descrição retrospectiva, inclusive empates. Não existia critério prévio para selecionar um vencedor, portanto não se declara “melhor modelo” validado. Nenhuma modalidade é superior automaticamente por maior escore. Bases diferem em tamanho, rótulos, unidade, recrutamento, coleta e validação. Não há pareamento nem late fusion.

## Instalação e dependências

Execute na raiz do projeto. O ambiente conferido usa Python 3.14; requisitos de análise em `requirements_clinicos.txt`: NumPy, pandas, SciPy, scikit-learn, Matplotlib e joblib. Não há dependências novas de geração Markdown. A disponibilidade de versões compatíveis deve ser verificada ao criar outro ambiente.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements_clinicos.txt
```

Nos comandos abaixo, `python` deve apontar para esse ambiente, ou use `.\.venv\Scripts\python`. Num ambiente já preparado, não reinstale dependências. JavaScript/node_modules não participam do pipeline. Notebook é opcional:

```powershell
python -m pip install "jupyterlab>=4,<5" "ipykernel>=6,<8"
python -m jupyterlab
```

Abra o notebook clínico na raiz e execute as células em ordem; ele chama o mesmo módulo da CLI. A execução pelo terminal não requer Jupyter, Colab ou upload interativo.

## Configuração dos caminhos dos dados

Os padrões correspondem às entradas reais da tabela de fontes. Para outras localizações, use argumentos explícitos, sem alterar código ou mover originais:

```powershell
python executar_tcc.py --clinica "CAMINHO/dados.csv" --proteomica "CAMINHO/tabela.txt" --genomica "CAMINHO/fastas" --metadados "CAMINHO/metadados_genomicos.csv" --resultados resultados --k 2
```

Os placeholders devem ser substituídos. `--k` aceita 1, 2 ou 3. Sem `--metadados`, o pipeline usa `metadados_genomicos.csv` dentro da pasta genômica. O CSV de metadados inicial usa UTF-8 com BOM e `;`; tabelas técnicas usam seus separadores registrados nos scripts.

## Como executar cada etapa

```powershell
python verificar_clinica.py
python verificar_proteomica.py
python -m unittest verificar_genomica verificar_relatorio -v
python clinica_pipeline.py --resultados resultados/avulsos/clinicos
python proteomica_pipeline.py --resultados resultados/avulsos/proteomicos
python genomica_pipeline.py --dados dados_genomicos --metadados dados_genomicos/metadados_genomicos.csv --saida resultados/avulsos/genomicos --k 2
```

CLIs avulsas criam pastas datadas e relatórios técnicos, mas não produzem manifesto conjunto; seus resultados não são misturados automaticamente no relatório integrado. Clínica aceita `--arquivo`, `--resultados`, `--folds` e `--semente`; proteômica aceita `--arquivo` e `--resultados`. Genômica aceita os argumentos acima e `--fasta ARQUIVO ...` para auditoria avulsa estrita sem integração.

O gerador inicial de metadados é opcional e recusa sobrescrever arquivos existentes:

```powershell
python gerar_metadados_genomicos.py --dados dados_genomicos --saida resultados/metadados_iniciais
```

Ele não substitui adjudicação clínica. Não regenere os metadados para perder alterações manuais.

## Como executar o pipeline completo

```powershell
python executar_tcc.py
```

Cria uma nova execução, roda as três modalidades viáveis e gera o relatório integrado. Falhas são registradas e as outras modalidades continuam. A síntese inclui falhas e resultados parciais da execução atual, com N/A e causa; o comando termina com erro se o pipeline ficar incompleto. A genômica exploratória concluída não significa classificador treinado.

Para **gerar novamente somente o relatório**, sem treinar modelos:

```powershell
python gerar_relatorio_integrado.py
```

Esse comando lê exclusivamente `resultados/execucao_atual.json`, nunca escolhe uma pasta pela maior data. Para uma execução explícita:

```powershell
python gerar_relatorio_integrado.py --execucao resultados/execucoes/ID_DA_EXECUCAO --resultados resultados
```

O identificador deve existir. Isso publica a execução solicitada como atual, sem mudar a origem dos artefatos. Somente manifestos do novo esquema com ID e hashes são aceitos; relatórios antigos permanecem preservados, mas não são promovidos silenciosamente ao novo formato. A cópia por execução é preservada: regeneração idêntica é permitida; conteúdo diferente exige outra execução.

## Onde encontrar os resultados finais

Relatório principal: [resultados/relatorio_final_integrado.md](resultados/relatorio_final_integrado.md). Tabela única: [resultados/tabela_comparativa.csv](resultados/tabela_comparativa.csv). Origem: [resultados/execucao_atual.json](resultados/execucao_atual.json).

Cada execução preserva sua síntese, tabela, manifesto, versões e diretórios técnicos. Clínica/proteômica: `comparacao_modelos.csv`, `metricas_folds.csv`, `previsoes_oof.csv`, matrizes CSV/PNG e relatórios. Clínica inclui distribuições, ROC por fold, associações, importâncias, coeficientes, auditorias e componente joblib. Proteômica inclui matriz, rótulos, inventário, QC e seleção por fold. Genômica inclui metadados integrados, auditoria, validação por arquivo, features, matriz exploratória, contagens, gráfico e configuração. O relatório contém links relativos para esses arquivos e gráficos.

## Limitações metodológicas e científicas

Suspeitas não confirmam PIF; controles informados não equivalem a negativos adjudicados. Amostra pequena, ausências diferenciais, recrutamento selecionado e possíveis recoletas limitam a clínica. Os experimentos ampliados podem revelar o processo diagnóstico. Pools proteômicos foram selecionados antes da CV, sem composição por gato, lotes e escala/zeros totalmente documentados. Métricas elevadas não comprovam diagnóstico individual.

Na genômica faltam diagnósticos, estudos, gatos independentes, orientação/completude e comparabilidade revisadas; há possíveis tecidos relacionados e divergências entre nomes/cabeçalhos. Editar nomes não resolve essas lacunas. Não há classificador, variantes posicionais ou métricas genômicas supervisionadas. Nenhuma modalidade tem teste externo, calibração ou demonstração de utilidade clínica. As diferenças entre modalidades não sustentam ranking científico.

## Reprodutibilidade

O manifesto registra ID, início UTC, status por etapa, versões, configuração, hashes do código e de cada artefato técnico. Caminhos de etapas ficam relativos à execução; o gerador rejeita caminhos fora dela, hashes alterados e resumos incoerentes com folds. Em etapas que falharam, artefatos parciais são preservados, mas não usados para aparentar sucesso. Não busca métricas antigas como substituição.

A versão principal é atualizada com a execução explicitamente indicada e sua cópia histórica é preservada. Não há exclusão automática de resultados. Dados originais e metadados não são sobrescritos. Os inventários de limpeza/preservação da revisão anterior permanecem históricos.

Verificação das saídas reais da base atual:

```powershell
python verificar_resultados.py --execucao resultados/execucoes/ID_DA_EXECUCAO
```

Recalcula métricas/matrizes, confere cobertura OOF e casas, compara a tabela integrada com a técnica, verifica links e confere os hashes das entradas registrados nesta execução. O verificador espera os diagnósticos genômicos não confirmados desta base e preserva a evidência já registrada, recusando sobrescrita. Exemplos artificiais existem somente em testes temporários de contrato, nunca nas análises científicas.

O argumento opcional `--inventario inventario_entradas_preservadas.json` compara adicionalmente as 777 entradas com a revisão histórica anterior. Ele deve falhar quando uma entrada tiver sido legitimamente editada depois daquele inventário: não o use como referência automática da versão atual dos dados. Nesta reorganização foi detectado que o CSV de metadados já havia mudado antes da tarefa; a versão atual foi preservada e seu hash está na configuração genômica da nova execução. O inventário histórico não foi reescrito.

## Referências bibliográficas e fontes dos dados

- Clínica: `06.2025_-_FIP_and_Housemate_Clinical_Data_Final_-_for_Dryad_Upload.csv`; referência bibliográfica/DOI e dicionário precisam ser preenchidos a partir da fonte original, sem inferência do nome.
- Proteômica: depósito **JPST003448**, tabela `21905517_merge-Metaboanalyst.txt`; [artigo associado na documentação histórica](https://www.nature.com/articles/s41598-025-03108-2). Esta reorganização não refez identificação/FDR ou coleta dos dados.
- Genômica: arquivos FASTA e metadados locais. Accessions e cabeçalhos ficam na auditoria; artigos, repositórios e diagnósticos individuais ainda precisam de documentação.
- Referências técnicas já utilizadas: [validação cruzada do scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html) e [importância por permutação](https://scikit-learn.org/stable/modules/permutation_importance.html).

As referências existentes foram preservadas; nenhuma nova fonte clínica ou bibliográfica foi inventada.
