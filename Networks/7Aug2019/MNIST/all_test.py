from all import Net

name = 'VGGNet'
input_width = 224
input_height = 224
input_channels = 1
num_classes = 10
learning_rate = 0.0001
momentum = 0.9
keep_prob = 0.5

my_class = Net(name,input_width, input_height, input_channels, num_classes, learning_rate,
                 momentum, keep_prob)

X_train,X_test,Y_train,Y_test= my_class.GetData()
pre_generator = my_class.pre_processing(X_train,Y_train)
model = my_class.VGG_model()
model = my_class.train(model,pre_generator,X_train,Y_train,X_test,Y_test)
Net.evaluate(model,X_test,Y_test)
