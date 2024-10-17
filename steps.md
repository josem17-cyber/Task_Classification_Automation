### Día 17/10/2024 HIDRAL 

- Hemos simplificado el código para que solo prediga la categoría y no el dueño que ya no tiene sentido.
- Hemos reducido el número de categorías a predecir, ya que no había suficientes datos en las demás categorías para que fuera efectivo
- Hemos creado un "classification_report" para ver la predicción de la categoría por tipo de tarea. Este ha sido el resultado:
    precision    recall  f1-score   support

           I       0.81      0.82      0.81      1305
        OMOD       0.83      0.78      0.81      2714
        OREP       0.80      0.86      0.83      2478
        PMOD       0.74      0.39      0.51      2023
        PREP       0.66      0.88      0.75      2586
         SAT       0.87      0.82      0.85      1314
         SEG       0.90      0.93      0.92      3196

    accuracy                           0.80     15616
   macro avg       0.80      0.78      0.78     15616
weighted avg       0.80      0.80      0.79     15616

**CONCLUSIONES**

- El funcionamiento general del modelo es aceptable en la mayoría de categorías. Hay que tener en cuenta la categoría PMOD para esto. ¿Puede ser que estos datos no sean reales totalmente?
- Los datos en algunas categorías no es suficiente, ya que hay muy pocos registros. Incluso puede que haya demasiados pocos registros para estas categorías, pero iremos viendo. Una opción es meter más años además de el 2023 que es con el que está funcionando el modelo.

**POR HACER**

- Word Embeddings -> Embeddings usando Transformers(BERT...) -> LLM, ChatGPT, etc...
- Separar los caracteres que no tengan una gran relevancia, sustituyendolas por espacios.
- Teniendo en cuenta el tema del idioma tendremos que comprobar primero cual es el idioma más común, a partir de esto usaremos StopWords en este lenguaje. Para ello podemos usar la librería *langdetect*. Aunque lo más lógico es que filtremos por idioma del cliente.
 