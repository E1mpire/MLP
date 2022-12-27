%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�����������
%�༶��2002011
%רҵ��������Ϣ����
%���ǰ�������磬���MINST��д�������ݿ��ļ�
%����㣺28*28����Ԫ�������㣺ʮ����Ԫ������㣺100����Ԫ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear,clc
%%�������ݵ�
input_layer=28*28;  %�����ϸ������
cell=50;  %������ϸ������
output_layer=2;  %�����ϸ������
n=4000;%���ݼ���С
t=1000; %���Լ���С
alpha=0.01;    %ѧϰЧ��
load('data.mat')
load('origin.mat')


trainimage=train_image(:,1:n);  %ѵ����
trainlabel=zeros(10,n);
for i=1:n
   trainlabel(train_label(i)+1,i)=1;
end

testimage=test_image(:,1:t);  %���Լ�
testlabel=zeros(10,t);
for i=1:t
   testlabel(test_label(i)+1,i)=1;
end

w1=w1o;%��һ���������Ȩֵ
w2=w2o;%�ڶ����������Ȩֵ
b1=b1o;%��һ���������ƫ��
b2=b2o;%�ڶ����������ƫ��



n1=w1*trainimage(:,1)+b1;
a1=LeakyReLU(n1);                     %��һ�㼤���LeakyRuLU
n2=w2*a1+b2;
a2=softmax(n2);                       %�ڶ��㼤���sigmoid

e(1)=sum(-trainlabel(:,1).*log(a2));             %��������ʧ����

%%%%%%%%%%%%%%%%%%%%%%

    for i=2:n
        s1=entrosoft(n2,trainlabel(:,i-1));   %����ڶ������ϵ��
        s2=(s1'*w2)'.*leaky(n1);                     %�����һ�����ϵ��
        w2=w2-alpha*s1*a1';
        w1=w1-alpha*s2*trainimage(:,i-1)';
        b1=b1-alpha*s2;
        b2=b2-alpha*s1;
        %����������ɣ���ʼ����
        n1=w1*trainimage(:,i)+b1;
        a1=LeakyReLU(n1);                   %��һ�㼤���LeakyRuLU
        n2=w2*a1+b2;
        a2=softmax(n2);                     %�ڶ��㼤���softmax
        e(end+1)=sum(-trainlabel(:,i).*log(a2)); %��������ʧ����
    end
plot(e)

%��ʼ��֤
error=0;
correct=0;
for i=1:t
    n1=w1*testimage(:,i)+b1;
    a1=LeakyReLU(n1);
    n2=w2*a1+b2;
    a2=softmax(n2);
    [~,posa]=max(a2);
    [~,posl]=max(testlabel(:,i));
    if posa~=posl    %����жϴ���
        error=error+1;
    else
        correct=correct+1;
    end
    if mod(i,20)==0
        fprintf('��ǰ����Ϊ%d,�ж�����Ϊ%d\n',posl-1,posa-1)
    end
end

disp(['>>ģ����ȷ��Ϊ',num2str(correct/(correct+error))])
function y=LeakyReLU(x)
%��gradΪFalse��ʱ���������ֵ����gradΪTrue��ʱ���������ֵ
    if x<0
        y=0;
    else
        y=x;
    end
end
function f=leaky(x)
%LeakyReLU�ĵ�����
x(x<0)=0;
x(x>=0)=1;
f=x;
end
function y=softmax(x)
%��gradΪFalse��ʱ���������ֵ����gradΪTrue��ʱ���������ֵ
    m=exp(x);
    y=m/sum(m);
end
function f=entrosoft(x,label)
%ֱ�Ӽ����s1�������˾Ϳ�https://blog.csdn.net/abc13526222160/article/details/84968161?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166918799816800180699635%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=166918799816800180699635&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-3-84968161-null-null.142^v66^control,201^v3^control_1,213^v2^t3_control1&utm_term=%E4%BA%A4%E5%8F%89%E7%86%B5%20%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E7%9A%84BP&spm=1018.2226.3001.4187
    m=exp(x);
    y=m/sum(m);
    f=y-label;
end
