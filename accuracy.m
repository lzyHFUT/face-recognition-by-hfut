% 面部识别程序

% 设置训练集和测试集的文件夹路径
trainFolderPath = 'E:\majority\face_recognition\train';  % 训练集文件夹路径
testFolderPath = 'E:\majority\face_recognition\test';    % 测试集文件夹路径

% 获取训练集文件夹下所有的jpg图片文件
trainImageFiles = dir(fullfile(trainFolderPath, '*.jpg'));

% 读取训练集图片并进行预处理
trainImageData = [];
for i = 1:length(trainImageFiles)
    imgPath = fullfile(trainFolderPath, trainImageFiles(i).name);
    img = imread(imgPath);
    if size(img, 3) == 3  % 检查是否为RGB图像
        img = rgb2gray(img);  % 转换为灰度图
    end
    img = imresize(img, [128 128]);  % 调整大小为128x128
    img = double(img(:));
    trainImageData = [trainImageData, img];
end
col_of_data = size(trainImageData, 2);

% 中心化 & 计算协方差矩阵
imgmean = mean(trainImageData, 2);
trainImageData = trainImageData - imgmean;
covMat = trainImageData' * trainImageData;
[COEFF, ~, explained] = pcacov(covMat);

% 获取测试集文件夹下所有的jpg图片文件
testImageFiles = dir(fullfile(testFolderPath, '*.jpg'));

% 读取测试集图片并进行预处理
testImageData = [];
for i = 1:length(testImageFiles)
    imgPath = fullfile(testFolderPath, testImageFiles(i).name);
    img = imread(imgPath);
    if size(img, 3) == 3  % 检查是否为RGB图像
        img = rgb2gray(img);  % 转换为灰度图
    end
    img = imresize(img, [128 128]);  % 调整大小为128x128
    img = double(img(:));
    testImageData = [testImageData, img];
end
col_of_test = size(testImageData, 2);
testImageData = testImageData - imgmean;  % 中心化

% 初始化变量
energy_percentages = 5:0.5:95;  % 从5%增长至95%
accuracies = zeros(1, length(energy_percentages));  % 存储不同能量百分比下的准确率

% 循环计算不同能量百分比下的识别准确率
for idx = 1:length(energy_percentages)
    percentage = energy_percentages(idx);
    
    % 选择构成指定能量百分比的特征值
    i = 1;
    proportion = 0;
    while proportion < percentage
        proportion = proportion + explained(i);
        i = i + 1;
    end
    p = i - 1;

    % 特征脸
    W = trainImageData * COEFF;    % N*M阶
    W = W(:, 1:p);                 % N*p阶

    % 训练样本在新坐标基下的表达矩阵 p*M
    reference = W' * trainImageData;

    % 随机选择30张测试图片计算正确率
    num_correct = 0;
    num_samples = 30;
    randIndices = randperm(col_of_test, num_samples);

    for j = 1:num_samples
        randIndex = randIndices(j);
        testImg = testImageData(:, randIndex);

        object = W' * testImg;
        distance = 1000000000000;
        for k = 1:col_of_data
            temp = norm(object - reference(:, k));
            if distance > temp
                aimone = k;
                distance = temp;
            end
        end

        % 提取图片名称中的序号
        testImgName = extractNumber(testImageFiles(randIndex).name);
        trainImgName = extractNumber(trainImageFiles(aimone).name);

        % 判断是否正确
        if testImgName == trainImgName
            num_correct = num_correct + 1;
        end
    end

    % 计算准确率
    accuracy = num_correct / num_samples;
    accuracies(idx) = accuracy;
    disp(['能量百分比 ', num2str(percentage), '% 时的分类器准确率: ', num2str(accuracy)]);
end

% 绘制准确率变化曲线
figure;
plot(energy_percentages, accuracies, 'b-o');
xlabel('能量百分比 (%)');
ylabel('识别准确率');
title('不同能量百分比下的识别准确率变化曲线');
grid on;

% 中心化函数
function cendata = center(testdata, meandata)
    for i = 1:size(testdata, 2)
        cendata(:, i) = testdata(:, i) - meandata;
    end
end

% 提取图片名称中的序号
function num = extractNumber(filename)
    % 使用正则表达式提取数字部分
    match = regexp(filename, '\d+', 'match');
    if ~isempty(match)
        num = str2double(match{1});
    else
        num = -1; % 如果没有匹配到数字，返回-1
    end
end