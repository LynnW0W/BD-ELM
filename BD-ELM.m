%% ʵ��Biased Dropout ELM�Լ�Biased DropConnect ELM
% clc; clear all;
%% ��ȡ���ݼ�,������ʼ��
%load My_Ionosphere.mat
load Diabetes.mat
Possibility= 0.7 ;                   %����Drop����
PossibilityHighGroup = 0.9;          %�����������������Ȩֵ��Drop����
PosiibilityLowGroup =  0.7;          %�����������������Ȩֵ��Drop����
HiddenNodes = 200;                   %������ڵ���
SwitchfoBiasedDropConnect = 0;       %�Ƿ����BiasedDropout��1Ϊ������0Ϊ�ر�
SwitchofBiasedDropout = 1;           %�Ƿ����BiasedDropout��1Ϊ������0Ϊ�ر�
% WeightPenaltyL2 = 1000;            %L2���򻯳ͷ�ϵ��

%% ʵ��BiasedDropELM
tic;
InputSamples = size(train_x,1);                            %��ȡ����������
InputDimension =  size(test_x,2);                          %��ȡ����ά��
InputWeight = rand(InputDimension, HiddenNodes)*2-1;       %�����������������Ȩֵ

%ʵ��Biased DropConnect ELM
if SwitchfoBiasedDropConnect == 1
    Mask = zeros(InputDimension,HiddenNodes);
    Threshold = median(InputWeight(:));                              %������Ȩֵ����ֵ��Ϊ��ֵ
    Location = find(InputWeight>=Threshold );
    Location(:,2) = binornd(1,PossibilityHighGroup,1,size(Location,1));   %������Ȩֵ���Ӹ߸���ֵ�Ĳ�Ŭ���ֲ�
    Mask(Location(:,1)) = Location(:,2);                                  %��������Ȩֵ��ֵ�ڸǾ���
    clear Location;
    Location = find(InputWeight<Threshold );
    Location(:,2) = binornd(1,PosiibilityLowGroup,1,size(Location,1));    %������Ȩֵ���ӵ͸���ֵ�Ĳ�Ŭ���ֲ�
    Mask(Location(:,1)) = Location(:,2);                                  %��������Ȩֵ��ֵ�ڸǾ���
    InputWeight = InputWeight.*Mask;                     %������Ȩֵ����Biased DropConnect
    clear Threshold;
    clear Location;
%     clear Mask
end

Bias = rand(1,HiddenNodes);                                %�������������ƫ��
TempH =  train_x*InputWeight + repmat(Bias,InputSamples,1);
H = sigm(TempH);                                           %�������������
clear TempH;

% ʵ��Biased Dropout ELM
if SwitchofBiasedDropout == 1
    Mask = zeros(InputSamples,HiddenNodes);                %��ʼ���ڸǾ���
    MeanH = mean(H);                                       %����������ڵ㼤��ֵ�ľ�ֵ
    Threshold = median(H);                                 %������ֵ��ֵ����ֵ��Ϊ��ֵ
    Location = find(MeanH>=Threshold );
    Location(2,:) = binornd(1,PossibilityHighGroup,size(Location,2),1);   %�߼���ֵ�ڵ�ֵ���Ӹ߸���ֵ�Ĳ�Ŭ���ֲ�
    Mask(:,Location(1,:)) = repmat(Location(2,:),InputSamples,1);
    clear Location;
    Location = find(H<Threshold);
    Location(2,:) = binornd(1,PossibilityHighGroup,size(Location,2),1);    %�ͼ���ֵ�ڵ�ֵ���ӵ͸���ֵ�Ĳ�Ŭ���ֲ�
    Mask(:,Location(1,:)) = repmat(Location(2,:),InputSamples,1);          %���ݼ���ֵ��ֵ�ڸǾ���
    H = H.*Mask;                                           %���������������Biased Dropout
    clear Threshold;
    clear Location;
    clear Mask
end
if size(H,1)>=size(H,2)
    OutputWeight = pinv(H' * H) * H' * train_y;
else
    OutputWeight = H' * (pinv(H * H') * train_y);
end
Time = toc;                                                %��ȡѵ��ʱ��
% ���ѵ��

%% ����BiasedDropELM
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