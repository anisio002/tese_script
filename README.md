# Utilização de Redes Informadas Pela Física no Preenchimento de Falhas em Séries de Temperatura do Ar

Neste estudo, um modelo baseado em Physics-Informed Neural Networks (PINN) foi desenvolvido e implementado com o objetivo de preencher falhas em dados micrometeorológicos e comparado com outros sete modelos. A função de custo do modelo foi derivada a partir de uma versão empírica da equação da difusão molecular. Os resultados foram comparados com sete modelos tradicionais de aprendizado de máquina, incluindo Regressão Linear, Regressão LASSO, Rede Elástica, KNN, Árvores de Classificação e Regressão (CART), Support Vector Regression (SVR) e Multi-Layer Perceptron (MLP).

Os resultados mostraram que os modelos baseados em redes neurais, incluindo o PINN, superaram a maioria dos outros modelos, com um erro médio quadrático (RMSE) médio de 0,08 °C em comparação com os valores médios de RMSE de 0,12 °C para os outros modelos. O PINN também exigiu menos épocas de treinamento em comparação com o MLP, com uma diferença máxima de apenas 62 épocas em cinco das oito regiões analisadas. No entanto, os modelos LASSO e EN apresentaram os maiores valores de RMSE, com uma média de aproximadamente 0,37 °C durante todas as fases e em todas as regiões analisadas. Esses resultados estiveram em conformidade com uma pesquisa anterior.

Além disso, foi observada a resiliência dos modelos em relação à proporção dos dados de treinamento, demonstrando sua capacidade de se adaptar a diferentes proporções de dados, com todos os modelos atendendo à quantidade mínima necessária para um aprendizado eficaz dos dados utilizados.

Em suma, este estudo destacou a eficácia dos modelos baseados em redes neurais, incluindo o PINN, na previsão de temperaturas em dados micrometeorológicos, proporcionando uma solução precisa para o preenchimento de lacunas. Esses resultados contribuem para o avanço das Ciências Ambientais e oferecem percepções para estudos futuros na área de preenchimento de dados faltantes, melhorando a integridade das análises climáticas e ambientais.

**Para maiores detalhes, visite a página do programa: [UFMT Pós Graduação em Física Ambiental](https://if.ufmt.br/instituto/site/)**

