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
