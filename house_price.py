from keras.datasets import boston_housing      
#add dataset
(train_data, train_targets), (
    test_data, test_targets)= boston_housing.load_data()

#print(train_data.shape)
#(404, 13)
#print(test_data.shape)
#(102, 13)

#Как видите, у нас имеются 404 обучающих и 102 контрольных образца, каждый 
#с 13 числовыми признаками, такими как уровень преступности, среднее число 
#комнат в доме, удаленность от центральных дорог и т. д.
#Цели — медианные значения цен на дома, занимаемые собственниками, в тысячах 
#долларов

#print( train_targets)
#[ 15.2, 42.3, 50. ... 19.4, 19.4, 29.1]
#Цены в основной массе находятся в диапазоне от 10 000 до 50 000 долларов США. 
#Если вам покажется, что это недорого, не забывайте, что это цены середины 1970-х 
#и в них не были внесены поправки на инфляцию

#данные сильно различаются их надо уместить в пределах единицы
#Нормализация данных:
mean = train_data.mean(axis=0) #average value
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std



from keras import models
from keras import layers
#for cross-validation
def build_model():
   model = models.Sequential() 
   model.add(layers.Dense(64, activation='relu',
   input_shape=(train_data.shape[1],)))
   model.add(layers.Dense(64, activation='relu'))
   model.add(layers.Dense(1))
   model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
   return model

import numpy as np
k = 4
num_val_samples = len(train_data) // k
num_epochs = 300
all_mae_histories = []
for i in range(k):
   print('processing fold #', i)
   val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
   
   val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
   
   partial_train_data = np.concatenate(    [train_data[:i * num_val_samples],
   train_data[(i + 1) * num_val_samples:]],   axis=0)

   partial_train_targets = np.concatenate(   [train_targets[:i * num_val_samples],
   train_targets[(i + 1) * num_val_samples:]],   axis=0)
   
   model = build_model() 
   
   history = model.fit(partial_train_data, partial_train_targets, validation_data=(
       val_data, val_targets),   epochs=num_epochs, batch_size=1, verbose=1)
   #print(history.history)
   mae_history = history.history['val_mae']
   all_mae_histories.append(mae_history)
   
   
   
#Создание истории последовательных средних оценок проверки 
#по K блокам
average_mae_history = [
 np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
   

import matplotlib.pyplot as plt
#Формирование графика с оценками проверок за исключением 
#первых 10 замеров
def smooth_curve(points, factor=0.9):
   smoothed_points = []
   for point in points:
       if smoothed_points:
           previous = smoothed_points[-1]
           smoothed_points.append(previous * factor + point * (1 - factor))
       else:
           smoothed_points.append(point)
   return smoothed_points
smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# test_mae_score = 2.594789981842041
# deviation from the norm +-2594$

model = build_model() 
model.fit(train_data, train_targets, 
 epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

   