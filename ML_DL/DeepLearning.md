# AI

- **ANI-artificial narrow intelligence**/ACHIEVED
  - smart speaker, self-drive car, web search, AI in farming or factories
- **AGI-artificial general intelligence**/MAYBE FUTURE
  - Do human can do

## Machine Learning

- **Supervised Learning**

  - **Give a input A then get a output B**

    spam filtering, machine translation, online ad, self-driving

  - Neural net give date progressing better performance than traditional AI

  - The larger is neural net the better is performance - **BIG DATA**

### Data

A table of data (**dataset**): input A

- Give the size of house and its price

  | size(feet^2^) | #bedrooms | price(1000$) |
  | :-----------: | :-------: | :----------: |
  |      523      |     1     |     115      |
  |      645      |     1     |     150      |
  |      708      |     2     |     210      |
  |     1034      |     3     |     280      |
  |     2290      |     4     |     355      |

  Use the size and number of bedrooms to its price to build a model to predict some other houses' prices

- **Acquiring data**

  - Manual labeling

    Tell AI the input and its output directly

  - From observing the behaviors

    give some behaviors data and build a prediction model

  - Download from web

- **Use and mis-use of data**
  - Collect data and use as soon as possible, then adjust the methods of collection, because if not, even a huge amount of data can be not valuable

- **Data is messy**

  - Data problems

    - Incorrect labels

    - Missing values

      | size(feet^2^) | #bedrooms | price(1000$) |
      | :-----------: | :-------: | :----------: |
      |      523      |     1     |     115      |
      |      645      |     1     |    0.001     |
      |      708      |  unknown  |     210      |
      |     1034      |     3     |   unknown    |
      |    unknown    |     4     |     355      |

      **structured data**

  - Multiple types of data

    images, audio, text(**unstructured**)

## Terminology

- Machine learning: use computer software that "gives computers the ability to learn without being explicitly programmed"
- Data science: extracting knowledge and insights from data
- Deep learning: build a neural network to give a input A then get a output B
- Other: Unsupervised learning, reinforcement leaning, graphical models, planning, knowledge graph ...

