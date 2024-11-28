Nicolas Alejandro Romero Montoya - 20201005035

Reto - 90 % de Acurrancy

- Inicialmente se ejecuta el código "crudo", este se deja correr normalmente. Por codigo, se limita el entrenamiento hasta 5 epocas (presentación de los datos completos 5 veces). El resultado es el esperado: 90 % de Acurracy.

- Se decide aumentar el numero de epocas a 20 con el fin de evaluar si el modelo le hacen falta más epocas (más entrenamineto) para obtener un mejor resultado. En la siguiente grafica se presenta el resultado obtenido:

  [Grafica](https://drive.google.com/file/d/1IHBH0sGvojEu6ZUMuCIL4J_OXfQSl2Mz/view?usp=sharing)
  
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

  - Se cambiaron las funciones de actvación a "Relu":
    - // First convolution layer
      model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        filters: 6,
        kernelSize: 5,
        activation: 'relu',
        padding: 'same'
      }));
      // Fully connected layers
      model.add(tf.layers.dense({ units: 120, activation: 'relu' }));
      model.add(tf.layers.dense({ units: 84, activation: 'relu' }));
      
    El comportamiento de la presición al 90% no cambió. Lo que indica que la limitación no está dada por la función de activación utilizada.
 
- Las capas de pooling están configuradas como "average pooling". Sin embargo, generalmente Max pooling es más efectivo que average pooling porque conserva las características más prominentes, como bordes y patrones clave, mientras que average pooling suaviza la información unicamente.

  - Se procede a setear las capas de pooling como "maxPooling"ya que puede mejorar la capacidad del modelo para generalizar y puede aumentar la precisión.
    - model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    El resultado se mantiene igual.

- La siguiente verificación fue añadir capas de dropout a la red, ya que estas nos ayudan a prevenir el sobreajuste al "apagar" aleatoriamente un porcentaje de neuronas durante el entrenamiento. Se optó por añadir 3 capas de dropout, una despues de cada capa convolucional y otra despues de la salida. Se pone un valor de 50% de dropout ya que es un valor tipicamente utilizado.
          - // Define the LeNet model
          function createLeNetModel() {
            const model = tf.sequential();
      
            // First convolution layer
            model.add(tf.layers.conv2d({
              inputShape: [28, 28, 1],
              filters: 6,
              kernelSize: 5,
              activation: 'relu',
              padding: 'same'
            }));
            dropout(0.5)
            model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
      
            // Second convolution layer
            model.add(tf.layers.conv2d({
              filters: 16,
              kernelSize: 5,
              activation: 'relu'
            }));
            dropout(0.5)
            model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
      
            // Flatten layer
            model.add(tf.layers.flatten());
      
            // Fully connected layers
            model.add(tf.layers.dense({ units: 120, activation: 'relu' }));
            model.add(tf.layers.dense({ units: 84, activation: 'relu' }));
            model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
            dropout(0.5)
            return model;
      Al poner un dropout del 50 %, el resultado no cambió. Este valor del dropout se varió entre el 20% y el 50% para las 3 capas y el resultado no fue distinto.

  - Por ultimo, se verifica el codigo inicial, y se verifica que el procesamiento de la infrmacion es correcto, ya que los datos de la imagen estan normalizados correctamente (se divide entre 255) y se leen las imagenes de la misma forma en la que están distribuidas originalemnte en el dataset.

En conclusión, se verificaron y modificaron convenientemente un buena cantidad de parámetros de la red convolucional con el fin de encontrar un mejor modelo. Sin embargo, el resultado siempre se mantuvo exactamente igual. Lo que indica que el problema está fundamentalmente en la base de datos. 
  



