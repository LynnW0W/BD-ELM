%% 实现Biased Dropout ELM以及Biased DropConnect ELM
% clc; clear all;
%% 读取数据集,参数初始化
%load My_Ionosphere.mat
load Diabetes.mat
Possibility= 0.7 ;                   %整体Drop概率
PossibilityHighGroup = 0.9;          %高隐含层输出或高输出权值的Drop概率
PosiibilityLowGroup =  0.7;          %低隐含层输出或低输出权值的Drop概率
HiddenNodes = 200;                   %隐含层节点数
SwitchfoBiasedDropConnect = 0;       %是否进行BiasedDropout，1为开启，0为关闭
SwitchofBiasedDropout = 1;           %是否进行BiasedDropout，1为开启，0为关闭
% WeightPenaltyL2 = 1000;            %L2正则化惩罚系数

%% 实现BiasedDropELM
tic;
InputSamples = size(train_x,1);                            %获取输入样本数
InputDimension =  size(test_x,2);                          %获取输入维度
InputWeight = rand(InputDimension, HiddenNodes)*2-1;       %随机生成隐含层输入权值

%实现Biased DropConnect ELM
if SwitchfoBiasedDropConnect == 1
    Mask = zeros(InputDimension,HiddenNodes);
    Threshold = median(InputWeight(:));                              %将连接权值的中值作为阈值
    Location = find(InputWeight>=Threshold );
    Location(:,2) = binornd(1,PossibilityHighGroup,1,size(Location,1));   %高连接权值服从高概率值的伯努利分布
    Mask(Location(:,1)) = Location(:,2);                                  %根据连接权值赋值掩盖矩阵
    clear Location;
    Location = find(InputWeight<Threshold );
    Location(:,2) = binornd(1,PosiibilityLowGroup,1,size(Location,1));    %低连接权值服从低概率值的伯努利分布
    Mask(Location(:,1)) = Location(:,2);                                  %根据连接权值赋值掩盖矩阵
    InputWeight = InputWeight.*Mask;                     %对连接权值进行Biased DropConnect
    clear Threshold;
    clear Location;
%     clear Mask
end

Bias = rand(1,HiddenNodes);                                %随机生成隐含层偏置
TempH =  train_x*InputWeight + repmat(Bias,InputSamples,1);
H = sigm(TempH);                                           %计算隐含出输出
clear TempH;

% 实现Biased Dropout ELM
if SwitchofBiasedDropout == 1
    Mask = zeros(InputSamples,HiddenNodes);                %初始化掩盖矩阵
    MeanH = mean(H);                                       %计算隐含层节点激活值的均值
    Threshold = median(H);                                 %将激活值均值的中值作为阈值
    Location = find(MeanH>=Threshold );
    Location(2,:) = binornd(1,PossibilityHighGroup,size(Location,2),1);   %高激活值节点值服从高概率值的伯努利分布
    Mask(:,Location(1,:)) = repmat(Location(2,:),InputSamples,1);
    clear Location;
    Location = find(H<Threshold);
    Location(2,:) = binornd(1,PossibilityHighGroup,size(Location,2),1);    %低激活值节点值服从低概率值的伯努利分布
    Mask(:,Location(1,:)) = repmat(Location(2,:),InputSamples,1);          %根据激活值赋值掩盖矩阵
    H = H.*Mask;                                           %对隐含层输出进行Biased Dropout
    clear Threshold;
    clear Location;
    clear Mask
end
if size(H,1)>=size(H,2)
    OutputWeight = pinv(H' * H) * H' * train_y;
else
    OutputWeight = H' * (pinv(H * H') * train_y);
end
Time = toc;                                                %获取训练时间
% 完成训练

%% 测试BiasedDropELM
tempH = test_x*InputWeight  + repmat(Bias,size(test_x,1),1);
HiddenOutput = sigm(tempH);
Yhat = HiddenOutput* OutputWeight;
% tempH = train_x*InputWeight  + repmat(Bias,size(train_x,1),1);
% HiddenOutput = sigm(tempH);
% Yhat = HiddenOutput* OutputWeight;
[~,argmax] = max(Yhat,[],2);
[~,amax] = max(test_y,[],2);
value_test = sum(argmax == amax)/length(amax);
% [~,argmax] = max(Yhat,[],2);
% [~,amax] = max(train_y,[],2);
% value_train = sum(argmax == amax)/length(amax);  