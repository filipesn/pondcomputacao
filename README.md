
# Ponderada de Computação

  

Este projeto é uma pipeline completa de visão computacional que lê uma imagem, usa visão computacional para traduzir ele em linhas, e com isso ele manipula uma tartaruga para desenhar o outline da imagem no Turtlesim com ROS2.

  

## Vídeo

  
Vídeo está direto no repositório.



## Arquitetura

  

O projeto atende aos requisitos de desenvolvimento e está dividido em dois módulos principais: a **Pipeline de Visão Computacional** e o **Controlador ROS 2**.

  

### 1. Visão Computacional (`cv_pipeline.py`)

  

Toda a extração foi desenvolvida de forma manual através das seguintes etapas de pré-processamento e detecção de bordas:

  

*  **Carregamento e Compresão:** A imagem é lida via OpenCV, convertida para RGB e comprimida com Average Pooling para reduzir o custo computacional.

*  **Conversão para Grayscale:** Transformação dos canais RGB para cinza usando os pesos tradicionais de luma para humanos.

*  **Otimização de Iluminação Local (CLAHE):** Implementação de Contrast Limited Adaptive Histogram Equalization para lidar com variações de iluminação, dinamicamente aumentando o contraste das bordas.

*  **Correção Gamma:** Ajuste para intesificar as áreas de interesse .

*  **Suavização Gaussiana:** Aplicação de convolução 2D com kernel Gaussiano para reduzir ruído.

*  **Filtro Sobel:** Aplicação de kernels Sobel ($K_x$ e $K_y$) com convolução 2D para calcular o gradiente de força e direção.

*  **Non-Maximum Suppression (NMS):** Afinamento das bordas.

*  **Limiarização por Histerese:** Classificação de pixels em bordas fortes e fracas, eliminando as fracas que não estão connectadas as fortes.

  

### 2. Planejamento e Controle (`turtle_controller.py`)

  

Mapeamento do contorno da imagem para Turtlesim:

  

*  **Extração de Caminhos:** Um algoritmo de busca rastreia as bordas contínuas gerando listas de coordenadas.

*  **Mapeamento para Turtlesim:** As coordenadas da imagem são convertidas e centralizadas para o plano cartesiano do Turtlesim.
 * Inscreve-se em `/turtle1/pose` para obter a posição da tartaruga.

* Publica velocidades e angulo em `/turtle1/cmd_vel`.

* Utiliza `/turtle1/set_pen` para levantar a caneta (`off=1`) ao transitar entre traços não conectados e baixar a caneta (`off=0`) enquanto desenha.
  
  
  

---

  

## Execução

  

### Pré-requisitos

  

* ROS 2 Humble

* Python 3 e as bibliotecas NumPy e OpenCV(para ler a imagem).

  

### Passos para Execução

  

1.  **Clone o repositório no seu workspace ROS 2:**

```bash
git clone <URL>
```

  

2.  **Coloque a imagem:**

Coloque a imagem do cachorro na raiz do seu ambiente.

3.  **Faça o build do pacote:**

```bash

colcon build turtle_cv_drawer
source install/setup.bash
```

4.  **Inicie o Turtlesim:**

```bash

ros2 run turtlesim turtlesim_node

```

5.  **Inicie o controlador (em outro terminal):**

```bash

source install/setup.bash
ros2 run turtle_cv_drawer turtle_controller
```

A tartaruga ira começar a desenhar automaticamente.  
 

---

  

## Desenvolvimento

Serei honesto, controlei mal meu tempo e o projeto não ficou na qualidade que eu desejo. Entretanto, está feito. Foi assim mais ou menos o desenolvimento:
Eu começei um blur gaussiano com um kernel 5x5, com uma limiarização global simples para determinar se um pixel devia ser preto ou branco. A imagem que foi providenciado pra esse projeto não reagiu muito bem com esse método já que é um cachorro preto, com metade da cabeça recebendo uma iluminação branca. Isso fez que ou você podia ter um lado esquerdo da imagem bem definido e com o direito entupido de lixo, ou você podia ter um lado direito bem definido mas completamente apagando o lado esquerdo. Não importa como eu ajustei, não consegui algo satisfatório que preservasse a estrutural facial do cachorro bem. A solução para isso foi o uso CLAHE, para dinamicamente ajustar o contraste para ter algo mais balanceado. Depois, fiz uma detecção de bordas típica com Sobel. O problema é que isso criava algumas bordas lixo, que foram eliminadas com o uso de histerese para determinar a força de cada linha. Esse ponto já estava demorando o processamento, então usei average pooling para compremir a imagem. 
Por fim, a implementação da tartaruga não mudou muito. Ele fui a ultima coisa desenvolvida. Ele basicamente olha pra imagem e entende ela como linhas, e descarta as muito pequenas. Depois ele simplesmente usa o sistema de nós do ROS para escutar a posição da tartaruga, mandar velocidade e ângulo, e controla a caneta sozinha.
Ficou assim porque eu primeiro desenvolvi tudo no google colab, e dai trouxe para o ambiente ROS no Ubuntu. Me dependi muito do uso de IA no desenvolvimento pois não controlei muito bem meu tempo, tive que balancear meus artefatos do projeto, a ponderada de elétrica, o desafio de matemática, e o questionamento do projeto e não dei prioridade suficiente a essa atividade.
