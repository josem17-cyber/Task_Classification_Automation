### Día 17/10/2024 HIDRAL 

- Hemos simplificado el código para que solo prediga la categoría y no el dueño que ya no tiene sentido.
- Hemos reducido el número de categorías a predecir, ya que no había suficientes datos en las demás categorías para que fuera efectivo
- Hemos creado un "classification_report" para ver la predicción de la categoría por tipo de tarea. Este ha sido el resultado:

| Clase                    | Precisión | Recall | F1-Score | Soporte |
| ------------------------ | --------- | ------ | -------- | ------- |
| I                        | 0.81      | 0.82   | 0.81     | 1305    |
| OMOD                     | 0.83      | 0.78   | 0.81     | 2714    |
| OREP                     | 0.80      | 0.86   | 0.83     | 2478    |
| PMOD                     | 0.74      | 0.39   | 0.51     | 2023    |
| PREP                     | 0.66      | 0.88   | 0.75     | 2586    |
| SAT                      | 0.87      | 0.82   | 0.85     | 1314    |
| SEG                      | 0.90      | 0.93   | 0.92     | 3196    |
| **Exactitud (accuracy)** |           |        | 0.80     | 15616   |
| **Promedio Macro**       | 0.80      | 0.78   | 0.78     | 15616   |
| **Promedio Ponderado**   | 0.80      | 0.80   | 0.79     | 15616   |


**CONCLUSIONES**

- El funcionamiento general del modelo es aceptable en la mayoría de categorías. Hay que tener en cuenta la categoría PMOD para esto. ¿Puede ser que estos datos no sean reales totalmente?
- Los datos en algunas categorías no es suficiente, ya que hay muy pocos registros. Incluso puede que haya demasiados pocos registros para estas categorías, pero iremos viendo. Una opción es meter más años además de el 2023 que es con el que está funcionando el modelo.

**POR HACER**

- Word Embeddings -> Embeddings usando Transformers(BERT...) -> LLM, ChatGPT, etc...
- Separar los caracteres que no tengan una gran relevancia, sustituyendolas por espacios.
- Teniendo en cuenta el tema del idioma tendremos que comprobar primero cual es el idioma más común, a partir de esto usaremos StopWords en este lenguaje. Para ello podemos usar la librería *langdetect*. Aunque lo más lógico es que filtremos por idioma del cliente.

### Día 18/10/2024 HIDRAL

- Vamos a probar en primer lugar a solo mantener los correos en los que el idioma es español ya que estos conforman la gran mayoría de ellos.

![Descripción de la imagen](media/descarga.png)

Abajo se muestra los resultados, al usar solo español, tenemos que aplicar STOPWORDS:

| Clase                    | Precisión | Recall | F1-Score | Soporte |
| ------------------------ | --------- | ------ | -------- | ------- |
| I                        | 0.86      | 0.66   | 0.75     | 757     |
| OMOD                     | 0.82      | 0.79   | 0.81     | 2008    |
| OREP                     | 0.79      | 0.87   | 0.83     | 1794    |
| PMOD                     | 0.78      | 0.32   | 0.45     | 1485    |
| PREP                     | 0.65      | 0.89   | 0.75     | 1982    |
| SAT                      | 0.80      | 0.82   | 0.81     | 762     |
| SEG                      | 0.87      | 0.95   | 0.91     | 2086    |
| **Exactitud (accuracy)** |           |        | 0.78     | 10874   |
| **Promedio Macro**       | 0.80      | 0.76   | 0.76     | 10874   |
| **Promedio Ponderado**   | 0.79      | 0.78   | 0.77     | 10874   |

**CONCLUSIONES**

- El cambio a solo idioma español no supone una gran diferencia debido a que para el modelo es lo mismo que esté un idioma que en otro. Vamos a mantener este cambio de **momento**

**POR HACER**

- Separar los caracteres que no tengan una gran relevancia, sustituyendolas por espacios.
  - Lo que está dentro de <> se puede elminar.
  - Hay que eliminar los '\t' y '\n'
- Word Embeddings -> Embeddings usando Transformers(BERT...) -> LLM, ChatGPT, etc...
- ¿Podríamos entrenar al modelo con únicamente el primer correo? A la hora de clasificar es esto lo que realmente ocurre, el modelo solo recibe un correo no el correo entero.

### Día 21/10/2024 HIDRAL

- Me he dado cuenta que tiene más sentido ahora mismo que nuestros datos únicamente muestren el primer correo, ya que al clasificar nuevas tareas es esto lo que tenemos que tener en cuenta. 
- Para ello hemos filtrado todos los datos en la consulta de la base de datos y después hemos agrupado por 'CodigoTarea', he elegido que siempre seleccione 'first', de esta manera en el caso del "Cuerpo" solo cogemos el primer cuerpo. 
- He podido observar que las precisiones son menores, debido probablemente a la falta de información dada. 
- No sé si realmente esta es la mejor opción, ya que considero que al final si yo entreno el modelo con la tarea completa y después le paso menos información de alguna manera estoy dificultando su comprension ya que fue entrenada con una tarea completa (la cual es más extensa). 
- Además el accuarcy no es del todo correcto ya que realmente la parte de test la estamos haciendo teniendo como entradas una tarea completa, esto no resulta coherente ya que es muy probable que la información que tenga que interpretar el modelo en el futuro únicamente contenga una frase. 
- Aquí podemos observar cual ha sido la predicción de nuestro modelo:
  
  | Clase                    | Precisión | Recall | F1-Score | Soporte |
  | ------------------------ | --------- | ------ | -------- | ------- |
  | I                        | 0.69      | 0.64   | 0.67     | 1486    |
  | OMOD                     | 0.82      | 0.58   | 0.68     | 4021    |
  | OREP                     | 0.69      | 0.83   | 0.75     | 3689    |
  | PMOD                     | 0.72      | 0.28   | 0.40     | 2999    |
  | PREP                     | 0.57      | 0.87   | 0.69     | 3499    |
  | SAT                      | 0.68      | 0.70   | 0.69     | 1386    |
  | SEG                      | 0.85      | 0.95   | 0.90     | 4306    |
  | **Exactitud (accuracy)** |           |        | 0.71     | 21386   |
  | **Promedio Macro**       | 0.72      | 0.69   | 0.68     | 21386   |
  | **Promedio Ponderado**   | 0.73      | 0.71   | 0.70     | 21386   |

**CONCLUSIONES**

- Lo más lógico es que el accuarcy fuera menor que en los casos anteriores, ya que tenemos menos cantidad de datos. Por otra parte tenemos que limpiar y vectorizar los textos, de esta manera conseguiremos una mejor precisión. Pero ya es un buen paso saber que nuestros datos deben estar de esta manera, ya que de la otra forma teniamos demasiados datos de test. 
- Por otra parte podemos barajar la opción de entrenar el modelo con las tareas completas y después pasar a test únicamente el primer correo para ver que tal trabaja cuando testea textos más cortos.

**POR HACER**

- Separar los caracteres que no tengan una gran relevancia, sustituyendolas por espacios.
  - Lo que está dentro de <> se puede elminar.
  - Hay que eliminar los '\t' y '\n'
- Word Embeddings -> Embeddings usando Transformers(BERT...) -> LLM, ChatGPT, etc...

**RECURSOS**

[Incrustaciones de palabras Tensorflow](https://www.tensorflow.org/text/guide/word_embeddings?hl=es-419)

[Una guía rápida para la limpieza de texto usando l](https://www.kaggle.com/code/edwight/una-gu-a-r-pida-para-la-limpieza-de-texto-usando-l#Removing-Stopwords)


### Día 22/10/2024 HIDRAL

- Lo que hemos realizado hoy básicamente ha sido la limpieza del texto como tal. Hemos conseguido limpiar el texto de tal manera que ahora se han eliminado los caracteres que son inncesarios para el modelo. Además de esto hemos conseguido quitar las StopWords, de esta manera mantenemos las palabras que sirven al modelo únicamente.
- Hemos empezado a trabajar en los Word Embeddings, creo que es un tema bastante profundo y que no se debe tomar a la ligera, pero podemos empezar por crear nuestros propios Embeddings, aunque creandolos en combinación con Word Embeddings genéricos.

**CONCLUSIONES**

- El modelo está bien preparado para que le apliquemos unos Embeddings a modo de preprocesamiento para la red neuronal de Keras. Actualmente el TF - IDF nos saca del paso pero sabemos que no es la mejor opción.
- Podemos observar que los datos en los PMOD y PREP no son los deseados, por lo que vamos a unificar todos los PMOD y PREP y llamarlos P para que de esta manera podamos sacar la conclusión correcta de si es capaz el modelo de saber que es un pedido o no.
- Estas han sido las precisiones del modelo para este caso, como podemos observar no han cambiado demasiado, esto es debido probablemente a que el modelo ya ha tomado los caracteres que hemos eliminado como innecesarios para su aprendizaje
  
  | Clase            | Precisión | Recall | F1-Score | Soporte |
  | ---------------- | --------- | ------ | -------- | ------- |
  | I                | 0.81      | 0.72   | 0.76     | 1486    |
  | OMOD             | 0.85      | 0.66   | 0.74     | 4021    |
  | OREP             | 0.71      | 0.88   | 0.79     | 3689    |
  | PMOD             | 0.73      | 0.34   | 0.46     | 2999    |
  | PREP             | 0.60      | 0.87   | 0.71     | 3499    |
  | SAT              | 0.78      | 0.77   | 0.77     | 1386    |
  | SEG              | 0.90      | 0.95   | 0.92     | 4306    |
  | **Exactitud**    |           |        | 0.76     | 21386   |
  | **Macro Avg**    | 0.77      | 0.74   | 0.74     | 21386   |
  | **Weighted Avg** | 0.77      | 0.76   | 0.74     | 21386   |


**POR HACER** 

- Llamar a las categorias PREP y PMOD como P, veremos la precisión en este caso
- Utilizar los datos de 2023 como entrenamiento y los datos de 2024 como test
- Word Embeddings (Genéricos + Creados) -> Embeddings usando Transformers(BERT...) -> LLM, ChatGPT, etc...

**RECURSOS**

[Incrustaciones de palabras Tensorflow](https://www.tensorflow.org/text/guide/word_embeddings?hl=es-419)

[Una guía rápida para la limpieza de texto usando l](https://www.kaggle.com/code/edwight/una-gu-a-r-pida-para-la-limpieza-de-texto-usando-l#Removing-Stopwords)

[How to Develop Word Embeddings in Python with Gensim](https://machinelearningmastery.com/develop-word-embeddings-python-gensim/)

[pypdf](https://pypdf.readthedocs.io/en/stable/)


### Día 23/10/2024 HIDRAL

- Con el fin de concretar la precisión entre Ofertas y Pedidos, he decidido unir los PREP y PMOD en una misma categoría llamada P, de esta manera podemos ver cual es la precisión del modelo a la hora de predecir si la tarea es un pedido o no, indistintamente de si es PREP o PMOD. Como vemos la precisión para los pedidos en general es bastante buena funcionando en el 95% de los casos, esto puede ser debido a que probablemente en todos estos indica adjunto pedido por lo que el modelo lo saca de manera correcta.

  | Clase        | precision | recall | f1-score | support |
  | ------------ | --------- | ------ | -------- | ------- |
  | I            | 0.74      | 0.68   | 0.71     | 1486    |
  | OMOD         | 0.75      | 0.75   | 0.75     | 4021    |
  | OREP         | 0.76      | 0.78   | 0.77     | 3689    |
  | P            | 0.96      | 0.95   | 0.95     | 6498    |
  | SAT          | 0.73      | 0.68   | 0.71     | 1386    |
  | SEG          | 0.91      | 0.94   | 0.92     | 4306    |
  | accuracy     |           |        | 0.85     | 21386   |
  | macro avg    | 0.81      | 0.80   | 0.80     | 21386   |
  | weighted avg | 0.85      | 0.85   | 0.85     | 21386   |

- Por otra parte he decidido también unir Ofertas, para ver si la precisión general en ellas era buena o no. Y resulta que esto es así, aunque habiamos conseguido una precisión de 74 o 75 entre OREP y OMOD, ahora tenemos casi un 90 % de aciertos en cuanto a las ofertas generales. 

  | Clase        | precision | recall | f1-score | support |
  | ------------ | --------- | ------ | -------- | ------- |
  | I            | 0.75      | 0.45   | 0.56     | 1486    |
  | O            | 0.88      | 0.95   | 0.91     | 7710    |
  | P            | 0.95      | 0.94   | 0.95     | 6498    |
  | SAT          | 0.60      | 0.65   | 0.63     | 1386    |
  | SEG          | 0.91      | 0.90   | 0.91     | 4306    |
  | accuracy     |           |        | 0.88     | 21386   |
  | macro avg    | 0.82      | 0.78   | 0.79     | 21386   |
  | weighted avg | 0.88      | 0.88   | 0.88     | 21386   |

- Al realizar el paso anterior ví que las SAT e I habían sido afectadas de manera negativa. Por lo que he decidido también englobarlas en una misma categoría aunque a priori puede no tener mucho sentido. Vemos que en general todas las métricas están por encima del 80% por lo que podríamos considerar un éxito este modelo, teniendo en cuenta que obviamente las categorías no nos sirven así directamente.

  | Clase        | precision | recall | f1-score | support |
  | ------------ | --------- | ------ | -------- | ------- |
  | I            | 0.80      | 0.81   | 0.80     | 2872    |
  | O            | 0.92      | 0.92   | 0.92     | 7710    |
  | P            | 0.92      | 0.96   | 0.94     | 6498    |
  | SEG          | 0.94      | 0.87   | 0.90     | 4306    |
  | accuracy     |           |        | 0.91     | 21386   |
  | macro avg    | 0.89      | 0.89   | 0.89     | 21386   |
  | weighted avg | 0.91      | 0.91   | 0.91     | 21386   |

- Hemos empezado a emplear **pypdf** para convertir a texto de los PDFs adjuntos, de esta manera los podríamos fusionar con la tarea (de pedidos en este caso) para que las clasifique debidamente (PREP o PMOD). A continuación, está el código que se emplea para leer cada PDF y convertirlo a texto:

  ``` python
  import pypdf

  reader = pypdf.PdfReader("../PDF/RD59SP0129_EN_26412745_715394_1078.pdf")
  page = reader.pages[0]
  print(page.extract_text(0))
  ```

- Para el preprocesamiento de nuestro modelo hemos empezado a aprender a crear nuestros propios **WordEmbeddings** utilizando **Gensim**. Para ello utilizaremos la clase **Word2Vec**, la cual funciona como veremos más abajo:
  
  ``` python
  from gensim.models import Word2Vec
  # define training data
  sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
               ['this', 'is', 'the', 'second', 'sentence'],
               ['yet', 'another', 'sentence'],
               ['one', 'more', 'sentence'],
               ['and', 'the', 'final', 'sentence']]
  # train model
  model = Word2Vec(sentences, min_count=1)
  # summarize the loaded model
  print(model)
  # summarize vocabulary
  words = list(model.wv.vocab)
  print(words)
  # access vector for one word
  print(model['sentence'])
  # save model
  model.save('model.bin')
  # load model
  new_model = Word2Vec.load('model.bin')
  print(new_model)
  ```

**CONCLUSIONES**

- El modelo funciona muy bien cuando se trata de predecir si el correo consiste en una oferta, pedido, seguimiento o incidencia. Sabemos que para que funcione en las categorías específicas tenemos que profundicar en el preprocesamiento.
- Una gran solución para los PREP y PMOD, es leer los documentos PDF, no sé que me dirá el equipo de informática de como acceder a estos documentos pero ya podemos ofrecer una solución para esto.
- Tenemos que seguir profundizando en la creación de Embeddings, pero teniendo en cuenta que lo más lógico es fusionarlo con uno que ya exista, lo veremos más adelante

**POR HACER**

- Word Embeddings (Genéricos + Creados) -> Embeddings usando Transformers(BERT...) -> LLM, ChatGPT, etc...

**CONCEPTOS**

- El significado de las métricas del **classification_report**:
  - *precision* : Porcentaje de predicciones positivas correctas en relación con el total de predicciones positivas
  - *recall* : Porcentaje de predicciones positivas correctas en relación con el total de positivos reales
  - *f1-score* : Media armónica ponderada de precision y recall. Cuanto más cercano es a uno mejor es el modelo
    $$
    F1 = \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
    $$

**RECURSOS**

[Incrustaciones de palabras Tensorflow](https://www.tensorflow.org/text/guide/word_embeddings?hl=es-419)

[Una guía rápida para la limpieza de texto usando l](https://www.kaggle.com/code/edwight/una-gu-a-r-pida-para-la-limpieza-de-texto-usando-l#Removing-Stopwords)

[How to Develop Word Embeddings in Python with Gensim](https://machinelearningmastery.com/develop-word-embeddings-python-gensim/)

[pypdf](https://pypdf.readthedocs.io/en/stable/)

### Implementación Detección SPAM (30/10/2024)

- Hemos obtenido de la base de datos los correos borrados, es decir todos aquellos que son SPAM, respuestas automáticas... Después lo hemos unificado a la base de datos de los correos que ya son tareas creadas.
- Esto va ser un primer paso a la implementación del modelo de clasificación de tal manera que podemos dictaminar si el correo es SPAM o no antes de clasificarlo
- Estos son los resultados del modelo en una primera instancia:

  | Clase            | precision | recall | f1-score | support |
  | ---------------- | --------- | ------ | -------- | ------- |
  | NO SPAM          | 0.95      | 0.98   | 0.96     | 3810    |
  | SPAM             | 0.97      | 0.93   | 0.95     | 2799    |
  | **accuracy**     |           |        | 0.96     | 6609    |
  | **macro avg**    | 0.96      | 0.95   | 0.96     | 6609    |
  | **weighted avg** | 0.96      | 0.96   | 0.96     | 6609    |
