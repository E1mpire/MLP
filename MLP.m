clear,clc
%%生成数据点
input_layer=28*28;  %输入层细胞个数
cell=50;  %隐含层细胞个数
output_layer=2;  %输出层细胞个数
n=4000;%数据集大小
t=1000; %测试集大小
alpha=0.01;    %学习效率
load('data.mat')
load('origin.mat')


trainimage=train_image(:,1:n);  %训练集
trainlabel=zeros(10,n);
for i=1:n
   trainlabel(train_label(i)+1,i)=1;
end

testimage=test_image(:,1:t);  %测试集
testlabel=zeros(10,t);
for i=1:t
   testlabel(test_label(i)+1,i)=1;
end

w1=w1o;%第一层神经网络的权值
w2=w2o;%第二层神经网络的权值
b1=b1o;%第一层神经网络的偏置
b2=b2o;%第二层神经网络的偏置



n1=w1*trainimage(:,1)+b1;
a1=LeakyReLU(n1);                     %第一层激活函数LeakyRuLU
n2=w2*a1+b2;
a2=softmax(n2);                       %第二层激活函数sigmoid

e(1)=sum(-trainlabel(:,1).*log(a2));             %交叉熵损失函数

%%%%%%%%%%%%%%%%%%%%%%

    for i=2:n
        s1=entrosoft(n2,trainlabel(:,i-1));   %计算第二层更改系数
        s2=(s1'*w2)'.*leaky(n1);                     %计算第一层更改系数
        w2=w2-alpha*s1*a1';
        w1=w1-alpha*s2*trainimage(:,i-1)';
        b1=b1-alpha*s2;
        b2=b2-alpha*s1;
        %参数更新完成，开始迭代
        n1=w1*trainimage(:,i)+b1;
        a1=LeakyReLU(n1);                   %第一层激活函数LeakyRuLU
        n2=w2*a1+b2;
        a2=softmax(n2);                     %第二层激活函数softmax
        e(end+1)=sum(-trainlabel(:,i).*log(a2)); %交叉熵损失函数
    end
plot(e)

%开始验证
error=0;
correct=0;
for i=1:t
    n1=w1*testimage(:,i)+b1;
    a1=LeakyReLU(n1);
    n2=w2*a1+b2;
    a2=softmax(n2);
    [~,posa]=max(a2);
    [~,posl]=max(testlabel(:,i));
    if posa~=posl    %如果判断错了
        error=error+1;
    else
        correct=correct+1;
    end
    if mod(i,20)==0
        fprintf('当前数据为%d,判断数据为%d\n',posl-1,posa-1)
    end
end

disp(['>>模型正确率为',num2str(correct/(correct+error))])
function y=LeakyReLU(x)
%当grad为False的时候输出函数值，当grad为True的时候输出导数值
    if x<0
        y=0;
    else
        y=x;
    end
end
function f=leaky(x)
%LeakyReLU的导函数
x(x<0)=0;
x(x>=0)=1;
f=x;
end
function y=softmax(x)
%当grad为False的时候输出函数值，当grad为True的时候输出导数值
    m=exp(x);
    y=m/sum(m);
end
function f=entrosoft(x,label)
%直接计算出s1，忘记了就看https://blog.csdn.net/abc13526222160/article/details/84968161?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166918799816800180699635%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=166918799816800180699635&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-3-84968161-null-null.142^v66^control,201^v3^control_1,213^v2^t3_control1&utm_term=%E4%BA%A4%E5%8F%89%E7%86%B5%20%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E7%9A%84BP&spm=1018.2226.3001.4187
    m=exp(x);
    y=m/sum(m);
    f=y-label;
end
