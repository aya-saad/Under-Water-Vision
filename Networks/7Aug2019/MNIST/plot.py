import keras
from matplotlib import pyplot as plt

history = model.fit(X_train, Y_train,validation_split = 0.1, epochs=50, batch_size=64)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
