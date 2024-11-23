# Análise de Vídeos com Reconhecimento de Emoções, Gestos e Posturas

Este script em Python processa vídeos para realizar detecções de emoções, gestos, posturas corporais e movimentos bruscos. Ele utiliza bibliotecas avançadas de visão computacional e aprendizado de máquina, como OpenCV, Mediapipe e DeepFace, para analisar quadros de vídeo e gerar relatórios detalhados sobre os resultados.

## Apresentação em Vídeo

- [Video de explicação do código e funcionalidades](https://youtu.be/oJmkRIkP3GQ)

## Funcionalidades

- **Detecção de emoções**: Reconhece emoções dominantes em rostos detectados.
- **Reconhecimento de gestos**: Identifica gestos específicos das mãos.
- **Análise de posturas corporais**: Detecta se uma pessoa está sentada, em pé ou em posições específicas como "T".
- **Identificação de movimentos bruscos**: Detecta e registra movimentos anômalos no vídeo.
- **Relatórios detalhados**: Gera relatórios em texto contendo estatísticas e informações relevantes do vídeo analisado.
- **Vídeos processados**: Salva o vídeo com as análises visuais incorporadas.

## Pré-requisitos

Antes de executar o script, instale as bibliotecas necessárias. Você pode fazer isso usando o comando:

```bash
pip install opencv-python mediapipe deepface tqdm

```

Além disso, é importante se certificar que tenha um vídeo no formato .mp4 para ser analisado.

### Privacidade

Importante ressaltar que o video analisado não será compartilhado com nenhuma entidade terceira, permanecendo apenas no computador de execução.

## Estrutura de Diretórios

Certifique-se de que a estrutura de diretórios esteja configurada da seguinte forma:

- script/
  - video_recognition.py
  - videos_input/
      - video.mp4
  - output/

- **videos_input/**: Contém o vídeo que será analisado.
- **output/**: Os relatórios e vídeos processados serão salvos aqui.

## Como Executar

1. Coloque o vídeo que deseja analisar na pasta **videos_input/** e renomeie-o como **video.mp4** (ou ajuste o nome no script, na variável **video_name**).
2. Execute o script com o seguinte comando:

``` bash
python video_recognition.py
```

3. Após a execução, os resultados serão salvos na pasta **output/**, incluindo:
- Um relatório em texto (**video_report.txt**) com as informações da análise.
- Um vídeo processado (**video_analyzed_video.mp4**) mostrando as detecções visuais.

## Otimização

O script pode ser configurado para processar todos os quadros ou apenas quadros em intervalos, utilizando o parâmetro **optimized** na função **detect_emotions_and_pose**. <br/>
 Para vídeos longos, defina **optimized=True** para melhorar o desempenho. <br/>
 Entranto, utilizando essa abordagem, a precisão da analise será menor.

## Observações

- O script é configurado para analisar um vídeo por vez. Para processar vários vídeos, ajuste a chamada da função detect_emotions_and_pose no final do script.

## Contato

Se tiver dúvidas ou sugestões, sinta-se à vontade para entrar em contato pelo e-mail **betinti@hotmail.com**