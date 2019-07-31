import combo
import os

base_dir = 'E:\keras\cats_and_dogs'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


train_generator,test_generator,validation_generator=combo.Net.pre_processing(train_dir,
                                                                             validation_dir,test_dir)
model = combo.Net.PlankNet_model()
model=combo.Net.train(model,train_generator,validation_generator)
combo.Net.evaluate(model,validation_generator,test_generator)
