clear,clc
%用来读取MNIST文件并生成mat文件
train_image=loadMNISTImages('train-images-idx3-ubyte');
train_label=loadMNISTLabels('train-labels-idx1-ubyte');

test_image=loadMNISTImages('t10k-images-idx3-ubyte');
test_label=loadMNISTLabels('t10k-labels-idx1-ubyte');

save('data.mat','train_image','train_label','test_image','test_label')