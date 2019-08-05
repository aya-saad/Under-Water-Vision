from combo_more import Net
import os

base_dir = 'E:\cats_and_dogs'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

name = 'OrgNet'
input_width = 150
input_height = 150
input_channels = 3
num_classes = 2
learning_rate = 0.001
momentum = 0.9
keep_prob = 0.8


my_class = Net(train_dir,validation_dir,test_dir,name,input_width, input_height, input_channels, num_classes, learning_rate,
                 momentum, keep_prob)

'''
train_generator,test_generator,validation_generator=combo.Net.pre_processing(train_dir,
                                                                             validation_dir,test_dir)
model = combo.Net.LeNet_model()
model=combo.Net.train(model,train_generator,validation_generator)
combo.Net.evaluate(model,validation_generator,test_generator)
'''
