Nicolas Alejandro Romero Montoya - 20201005035

Reto - 90 % de Acurrancy

- Inicialmente se ejecuta el código "crudo", este se deja correr normalmente. Por codigo, se limita el entrenamiento hasta 5 epocas (presentación de los datos completos 5 veces). El resultado es el esperado: 90 % de Acurracy.

- Se decide aumentar el numero de epocas a 20 con el fin de evaluar si el modelo le hacen falta más epocas (más entrenamineto) para obtener un mejor resultado. En la siguiente grafica se presenta el resultado obtenido:

  ----------------------------------------------grafica------------------
  
- Debido a que el resultado obtenido es exactamente igual al obtenido en la primera prueba se procedió a empezar a suprimir capas existentes de la red neuronal. El comportaminto debería cambiar, asi fuese de foroma mínima, al empezar a suprimir capas. Las capas originales de la red neuronal se ilustran en el siguiente codigo resumido:

    function createLeNetModel() {
      const model = tf.sequential();
      model.add(tf.layers.conv2d({ inputShape: [28, 28, 1], filters: 20, kernelSize: 5, activation: 'tanh', padding: 'same' }));
      model.add(tf.layers.averagePooling2d({ poolSize: 2, strides: 2 }));
      model.add(tf.layers.conv2d({ filters: 16, kernelSize: 5, activation: 'tanh' }));
      model.add(tf.layers.averagePooling2d({ poolSize: 2, strides: 2 }));
      model.add(tf.layers.flatten());
      model.add(tf.layers.dense({ units: 150, activation: 'tanh' }));
      model.add(tf.layers.dense({ units: 56, activation: 'tanh' }));
      model.add(tf.layers.dense({ units: 10, activation: 'softmax' })); // Corrected to match number of classes
  
  La capas suprimidas fueron las siguientes, en orden secuencial:

  1. Se suprimió la capa densa intermedia (model.add(tf.layers.dense({ units: 56, activation: 'tanh' }));). Se obtuvo exactamente el mismo resultado.
  2. Se suprimió la primera capa densa (model.add(tf.layers.dense({ units: 150, activation: 'tanh' }));). Se obtuvo exactamente el mismo resultado.
  3. Se suprimió el primr pooling (model.add(tf.layers.averagePooling2d({ poolSize: 2, strides: 2 }))) y la segunda capa convolucional ( model.add(tf.layers.conv2d({ filters: 16, kernelSize: 5, activation: 'tanh' }))). Se obtuvo exactamente el mismo resultado.
 
Se evidencia que a pesar de que se llega a un modelo muchísimo más simple que el modelo inicial, el resultado a nivel de metricas de rendimiento es el mismo a nivel de preisión. A nivel de las metricas "loss" en entrenamiento y validación, estas si presentaron cambios leves conforme se variaba el modelo. En algunos casos la metrica "loss" en entrenamiento y validación tienden a converger al mismo valor más rapidamente (menos epocas para converger al mismo valor). Sin embargo, la presición, que es el valor que más se desea sea cercano a uno, no se modificó. Esto indica que la limitación del rendimiento de la red neuronal no se debe a la complejidad de la misma. 

    - Dado lo anterior, se intuyen 2 cosas:
      - El problema puede deberse a una mala presentación de los datso (imagenes) y por ende no es posible obtener un mejor resultado.
      - El problema puede deberse factores como las capas de pooling, los parámetros de entrenamiento, las funciones de activación o un posible preprocesamiento de losdatso erroneo o incompleto.

- Se verifican las funciones de activación utilizadas: para las capas de entrada e intermedias, se tiene la función de activación tangente hiperbolica. La literatura indica que esta función puede limitar el "performance" de estas redes neuronales en muchos casos. Normalmente se utiliza la función Relu, ya que generalente generaliza mejor para aplicaciones en deep learning.

  - Se cambiaron las funciones de actvación a "Relu" y el comportamiento de la presición al 90% no cambió. Lo que indica que la limitación no está dada por la función de activación utilizada.
