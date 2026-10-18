# Modalidade clínica do TCC

Esta etapa pertence ao TCC “Integração multimodal de dados genômicos e clínicos via Machine Learning para predição de virulência da Peritonite Infecciosa Felina (PIF)”. Ela audita os dados clínicos, constrói variáveis e compara modelos de classificação. Pode ser instalada, executada e interpretada somente com este documento; não depende da execução da etapa genômica ou da leitura de outro README.

Esta etapa classifica **Healthy housemate (0)** versus **casos incluídos no estudo (1)**. As categorias de suspeita não confirmam PIF. O CSV não fornece alvo de virulência; portanto esta análise não valida a predição de virulência proposta no tema geral do TCC.

A entrada padrão é `06.2025_-_FIP_and_Housemate_Clinical_Data_Final_-_for_Dryad_Upload.csv`, com 100 registros e 12 colunas. Na execução disponível, 28 registros foram excluídos conforme os critérios existentes, restando 72 animais de 43 casas: 43 controles e 29 casos incluídos. `FIPID` identifica o registro do animal; `Home number ID` identifica a casa para agrupamento da validação. Identificadores não são usados como preditores.

## Arquivos

- `Pif_Dados_Clinicos_Reprodutivel.ipynb`: notebook clínico, com etapas numeradas e explicações.
- `clinica_pipeline.py`: funções de auditoria, engenharia, validação e interface clínica reutilizável.
- `requirements_clinicos.txt`: dependências com faixas compatíveis. A execução registra versões exatas; `requirements_clinicos_executado.txt` registra o ambiente usado na entrega.
- `verificar_clinica.py`: verifica agrupamento, alinhamento, datas inválidas, categorias inesperadas, serialização e previsão com raça inédita.
- `resultados_clinicos/execucao_.../`: uma pasta nova por execução, com tabelas, gráficos, configurações, relatório e componente serializado.

O CSV de entrada não é sobrescrito. A entrada padrão é o CSV existente na pasta; informe outro caminho por configuração ou pela CLI.

## Instalação e execução

É necessário Python com `venv` e `pip`. As dependências declaradas incluem NumPy, pandas, SciPy, scikit-learn, Matplotlib, joblib e ferramentas Jupyter. No terminal, a partir da raiz do projeto, com o CSV de entrada disponível:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements_clinicos.txt
.\.venv\Scripts\python clinica_pipeline.py
```

O comando cria uma nova pasta `resultados_clinicos/execucao_.../`, sem sobrescrever execuções anteriores. As pastas `resultados*/` são ignoradas pelo Git; em uma cópia nova do repositório é necessário gerar os resultados ou obter uma execução existente. Não é necessário ter arquivos genômicos para executar a clínica.

Para outro arquivo ou configuração:

```powershell
.\.venv\Scripts\python clinica_pipeline.py --arquivo "CAMINHO\dados.csv" --folds 5 --semente 42
```

Para o notebook:

```powershell
.\.venv\Scripts\python -m jupyterlab
```

Abra `Pif_Dados_Clinicos_Reprodutivel.ipynb`, selecione o kernel da `.venv` e execute as células em ordem. O diretório de trabalho deve conter o módulo; o caminho do CSV é configurável na primeira célula. O notebook não usa Google Colab nem upload interativo. A CLI executa o mesmo código, sem depender de Jupyter. Importações podem demorar na primeira execução.

Verificações de integridade: `python verificar_clinica.py`. Para executar pelo terminal, use `python clinica_pipeline.py`; para executar o notebook e salvar suas saídas, abra-o no Jupyter e execute as células em ordem.

Os scripts auxiliares de geração e execução do notebook foram removidos por serem opcionais. O notebook existente e a execução direta do pipeline permanecem disponíveis, sem alterações na análise clínica.

## Decisões metodológicas

1. Leitura como texto preserva `NA`, `unk` e a grafia original. Uma tabela explicita normalização de espaços/caixa; `N/No` e `not perfomed/not performed` têm equivalências documentadas nas funções. Nenhuma alteração é gravada no CSV.
2. Somente os rótulos de exclusão conhecidos são retirados; categorias novas interrompem a execução. A combinação grupo/diagnóstico também é validada. Os 29 incluídos têm suspeitas diagnósticas, sem confirmação presumida.
3. FIPID serve para alinhamento. Duplicação ou ausência interrompe a modelagem para investigação. `Home number ID` determina os grupos da validação; animais da mesma casa não aparecem em treino e validação no mesmo fold. Há estratificação aproximada e checagem de duas classes em ambas as partições; reduz-se o número de folds caso necessário.
4. Idade é calculada com datas estritas mês/dia/ano, formato sustentado por datas inequívocas no arquivo. Datas inválidas/futuras e idades negativas ou acima do limite configurado são auditadas e resultam em idade ausente. Nenhuma linha é descartada por ausência. Sexo e raça permanecem categóricos; nomes de raças diferentes não são fundidos sem dicionário clínico.
5. Experimentos: demográfico (idade, sexo, raça); demográfico + forma; anterior + resultados qualitativos de exames; anterior + tratamento. O conjunto demográfico é a referência de menor exposição ao processo diagnóstico, mas ainda pode sofrer seleção e ausência diferencial. Forma, realização dos exames e tratamento são fontes fortes de vazamento semântico. Os experimentos ampliados são retrospectivos e exploratórios. Não há garantia de disponibilidade temporal desses dados antes da suspeita diagnóstica.
6. Exames são reduzidos a positivo, negativo, não realizado ou ausente; títulos, datas e material não são usados para evitar fragmentação por amostra e mistura de escalas laboratoriais. Isso perde detalhe e não torna o experimento prospectivamente válido. NA de forma ou tratamento não é interpretado como forma ausente ou animal não tratado.
7. Pipeline aprende imputação numérica por mediana, padronização e categorias OneHot apenas no treino. Ausência categórica é uma categoria explícita. Categorias inéditas são ignoradas pelo codificador. O valor zero operacional para coluna numérica totalmente ausente no treino é mantido pelo `keep_empty_features`; não representa idade real. Nomes codificados e medianas por fold são exportados.
8. Dummy, regressão logística, SVM linear e Random Forest têm hiperparâmetros fixados antes da avaliação. Não há busca, seleção de atributos, escolha automática do vencedor nem calibração. Modelos de aprendizado usam ponderação das classes. SVM usa margem para ROC-AUC; os demais produzem probabilidades não calibradas.
9. Um fold é uma parte reservada para validação cruzada. Previsões OOF (out-of-fold) são produzidas, para cada animal, por um modelo treinado sem o fold desse animal. Métricas são calculadas por fold e resumidas por média e desvio-padrão amostral. O desvio não é intervalo de confiança. As previsões OOF mantêm FIPID, grupo, rótulo, fold e tipo de escore, permitindo investigar falsos positivos e falsos negativos. As matrizes são OOF. Não se calcula AUC conjunta de margens de SVM de folds distintos.
10. As associações usam permutação restrita dentro da casa, com diferença média de idade ou estatística qui-quadrado; não dependem da aproximação assintótica para células raras. O p empírico usa correção +1. Benjamini–Hochberg é aplicado à família de testes exportada. Essa estratégia exige permutabilidade condicional e não elimina seleção; com poucas casas de duas classes, há pouca informação. V de Cramér é descritivo e tende a ser otimista com muitas categorias. A inclusão de ausências no teste categórico mede também o processo de coleta, não só biologia. A correção BH não garante controle sob qualquer dependência entre testes; todos os resultados são exploratórios.
11. Importância por permutação é calculada em cada validação, com queda de ROC-AUC por variável original. Repetições não são amostras independentes; o resumo primeiro calcula a média dentro do fold e depois a dispersão entre folds. Permutar individualmente rompe correlação intradomiciliar e entre atributos: trate a importância como sensibilidade do modelo, sem inferência causal. Coeficientes OneHot de logística/SVM são exportados por categoria/fold; categorias ausentes em certos treinos não possuem coeficiente naquele fold. Sinais positivos favorecem a classe de incluídos.

Referências oficiais: [validação cruzada e grupos](https://scikit-learn.org/stable/modules/cross_validation.html), [importância por permutação](https://scikit-learn.org/stable/modules/permutation_importance.html).

## Reutilização do componente

```python
import joblib
from clinica_pipeline import carregar_dados, Configuracao

componente = joblib.load("resultados_clinicos/execucao_.../componente_demografico_logistica.joblib")
novos = carregar_dados(Configuracao(arquivo="outros_dados.csv"))
previsoes = componente.prever(novos)  # FIPID separado, ordem original, previsão e probabilidade
```

O método `prever` recebe linhas clínicas com os nomes originais; não exige alvo, grupo nem categoria diagnóstica. Os campos utilizados na engenharia e o FIPID precisam existir. `carregar_dados` valida o esquema integral de pesquisa; para previsão, também é possível passar diretamente um DataFrame compatível. Os códigos de sexo/exames/tratamento desconhecidos exigem revisão; raça/forma inéditas são aceitas pelo OneHot. O componente demográfico de referência é ajustado em todos os dados após a avaliação. Não atribua a esse ajuste uma avaliação independente e carregue apenas arquivos joblib confiáveis.

Para integração com outra modalidade, preservar correspondência validada entre ID de animal, amostra, casa e momento de coleta; proveniência; definição do desfecho; disponibilidade temporal; critérios de inclusão; incerteza diagnóstica; partições comuns e regras de acesso/consentimento. O módulo clínico não executa genômica ou fusão. Separadamente, `multimodal_pipeline.py` audita FASTA e disponibiliza uma função de Late Fusion, que combina probabilidades dos modelos por amostra. Nos dados atuais há apenas um genoma sem vínculo documental aos animais; portanto não existe avaliação multimodal validada. Isso não impede a execução clínica independente.

O ponto de saída para integração é `previsoes_oof.csv`, com `FIPID`, `alvo`, `experimento`, `modelo`, `fold`, `previsao`, `escore` e `tipo_escore`, além dos grupos e campos de auditoria. A regressão logística demográfica é a referência pré-especificada: seu `escore` é uma probabilidade não calibrada. Para uma comparação multimodal válida, exigir correspondências documentadas, alvos iguais e divisões de treino/validação comuns; o número de fold sozinho não comprova compatibilidade. Não utilizar previsões do modelo final sobre seus próprios dados de ajuste como avaliação independente.

## Resultados e limitações

Cada execução gera `RELATORIO.md` com números realmente calculados e comparação completa. Consulte também `problemas_datas.csv`, `ausentes_atributos.csv`, `associacoes_exploratorias.csv`, `erros_oof.csv`, `frequencia_erros_amostra.csv` e `importancia_validacao_resumo.csv`.

As tabelas principais são `comparacao_modelos.csv` (resumo das métricas), `metricas_folds.csv` (avaliação por fold), `divisao_folds.csv` (amostras e partições) e `previsoes_oof.csv` (previsões individuais). O arquivo `configuracao_versoes.json` registra parâmetros, versões e hashes; `componente_demografico_logistica.joblib` contém o componente final reutilizável.

Na execução `resultados_clinicos/execucao_20261018T170043_012603Z`, a regressão logística demográfica apresentou ROC-AUC de 0,768 ± 0,148 e F1 de 0,677 ± 0,167, como média ± desvio-padrão entre cinco folds. Esses valores avaliam o alvo de inclusão no estudo na coorte clínica original; não representam virulência nem desempenho de fusão.

Os principais limites são o pequeno número de animais/casas, suspeitas diagnósticas, seleção dos grupos, possíveis recoletas com IDs diferentes, ausência diferencial e falta de validação externa. A descrição de recoleta em um registro excluído precisa ser conferida na documentação original. O rótulo de casa é assumido como identificador de domicílio; confirme isso no dicionário do estudo antes de publicar. Não se concluem causalidade, biomarcadores, desempenho clínico validado ou virulência. Desempenhos elevados com informação diagnóstica requerem investigação de vazamento, e não celebração como confirmação clínica.
