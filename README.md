# FormUp - Deadlift Form MVP

Este projeto é o MVP do FormUp, orientado à análise do levantamento terra
(deadlift).

```text
vídeo/webcam -> MediaPipe Pose -> 21 características -> GRU -> deadlift
                                      |
                                      -> regras de técnica e repetições
```

## O que o MVP faz

- aceita webcam ou arquivo de vídeo;
- detecta os landmarks corporais com MediaPipe;
- suaviza a pose para reduzir ruído;
- reconhece a sequência temporal com o modelo GRU existente;
- diferencia deadlift de outras atividades;
- identifica três variações populares:
  - levantamento terra convencional;
  - levantamento terra sumô;
  - levantamento terra romeno (RDL);
- conta repetições por uma máquina de estados;
- exibe feedback específico de postura;
- não classifica os demais exercícios como se fossem deadlift.

O modelo legado ainda possui classes adicionais (`squat`, `bench press` e
outras). Elas continuam armazenadas no artefato treinado para preservar a
compatibilidade, mas o runtime FormUp usa somente a classe `deadlift`. As
variações não são classes do modelo legado: são inferidas por regras
geométricas sobre a pose detectada.

## Exercícios no documento FormUp

O documento descreve um MVP focado em **um exercício: o levantamento terra**.
Ele contextualiza os três movimentos do powerlifting (agachamento, supino e
levantamento terra), mas não propõe implementar os três no MVP. Esta versão
mantém o foco no deadlift e adiciona três variações do próprio movimento.

## Execução

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute com a webcam padrão:

```bash
python Real_Time_Inference.py
```

Execute um vídeo gravado:

```bash
python Real_Time_Inference.py --source "C:\caminho\para\deadlift.mp4"
```

Pressione `q` para sair.

## Organização técnica

- [`pose_features.py`](pose_features.py): definições compartilhadas dos
  landmarks, ângulos, distâncias e 21 características do modelo.
- [`MediaPipe.py`](MediaPipe.py): extração dos landmarks de datasets de vídeos.
- [`Feature_Engineering.py`](Feature_Engineering.py): geração offline das
  características usando o mesmo módulo do runtime.
- [`checkRules.py`](checkRules.py): identificação das variações, máquina de
  estados e feedback do deadlift.
- [`Real_Time_Inference.py`](Real_Time_Inference.py): aplicação FormUp para
  webcam ou vídeo.
- `files/GRU_model_7/`: modelo, scaler e mapeamento de rótulos usados no
  runtime.

## Limitações atuais

O modelo de reconhecimento foi treinado originalmente para várias classes e
não para variações ou erros técnicos específicos. Portanto, a confiança da GRU
indica somente que a sequência parece ser deadlift; a variação e a qualidade
da técnica vêm das regras geométricas. A classificação de variações deve ser
validada com vídeos rotulados por especialistas antes de ser usada como
avaliação esportiva.
