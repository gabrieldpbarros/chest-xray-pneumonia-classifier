# Desenvolvimento de modelos de redes neurais convolucionais para o diagnóstico de casos de pneumonia infantil através de imagens de radiografias torácicas.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red)

**Discentes**: Davi Santos, Felipi Martins, Gabriel Delgado

> [!NOTE]
> Este projeto é um refinamento de um projeto da UC de Redes Neurais.
>
> Link do repositório: https://github.com/gabrieldpbarros/Artificial-Neural-Networks/tree/main/Projeto_Final

## Resumo
Através da implementação de técnicas de visão computacional, como segmentação de imagens, Grad-CAM e Redes Neurais Convolucionais, busca-se desenvolver um modelo de inteligência artificial capaz de identificar e classificar precisamente casos de pneumonia em imagens de radiografias infantis. Para isso, foi aplicada uma metodologia de aprendizado transferido em modelos pré-treinados do projeto anterior, de forma a acelerar o treinamento dos novas redes, submetidas às entradas cuidadosamente preparadas por meio de técnicas de processamento de imagens, conforme convencionado para o estudo de imagens médicas.

## O que o projeto aprimora?
Anteriormente, apenas diferentes topologias de CNNs foram testadas no [dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), atingindo os seguintes resultados:

<div align="center" style="display: flex; justify-content: center;">
    <table style="border-collapse: collapse; text-align: left;">
        <thead>
            <tr style="border-bottom: 2px solid var(--background-modifier-border)';">
                <th style ="padding: 10px 20px;">Modelo</th>
                <th style ="padding: 10px 20px;">Acurácia</th>
                <th style ="padding: 10px 20px;">Recall</th>
                <th style ="padding: 10px 20px;">F1-Score</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid var(--background-modifier-border);">
                <td style="padding: 10px 20px;">Baseline</td>
                <td style="padding: 10px 20px;">94.54%</td>
                <td style="padding: 10px 20px;">95.56%</td>
                <td style="padding: 10px 20px;">96.24%</td>
            </tr>
            <tr>
                <td style="padding: 10px 20px;">Proposto</td>
                <td style="padding: 10px 20px;">90.96%</td>
                <td style="padding: 10px 20px;">92.76%</td>
                <td style="padding: 10px 20px;">93.74%</td>
            </tr>
        </tbody>
    </table>
</div>

Contudo, é possível que a rede tenha destacado regiões que não são de interesse para análise. Isso seria verificável caso fosse aplicada uma técnica de explicabilidade, como um mapa de ativação por Grad-CAM.

Dessa forma, a primeira ideia parte da premissa de visualizar que regiões das imagens os modelos treinados estão valorizando para a análise. Assim, podemos certificar a validez do resultado fornecido pela rede: se a classificação de fato vem da observação de uma região no pulmão, então de fato a rede possui uma aplicação prática.

Além disso, no campo do diagnóstico por imagem, a etapa de pré-processamento de imagem possui mais etapas do que as implementadas no projeto anterior, como segmentação e aplicação de filtros. De modo a aproximar o projeto à literatura, o pré-processamento adequado das imagens do dataset também foi adicionado.