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

% 选择构成95%能量的特征值
i = 1;
proportion = 0;
while proportion < 95
    proportion = proportion + explained(i);
    i = i + 1;
end
p = i - 1;

% 特征脸
W = trainImageData * COEFF;    % N*M阶
W = W(:, 1:p);                 % N*p阶

% 训练样本在新坐标基下的表达矩阵 p*M
reference = W' * trainImageData;

% 展示平均脸
averageFace = reshape(imgmean, [128, 128]);
figure;
imshow(averageFace, []);
title('平均脸');

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
disp(['分类器准确率: ', num2str(accuracy)]);

% 选择特定图片进行测试并展示结果
specificTestImgPath = 'E:\majority\face_recognition\lzy2.jpg';
specificTestImg = imread(specificTestImgPath);
if size(specificTestImg, 3) == 3  % 检查是否为RGB图像
    specificTestImg = rgb2gray(specificTestImg);  % 转换为灰度图
end
specificTestImg = imresize(specificTestImg, [128 128]);  % 调整大小为128x128
specificTestImg = double(specificTestImg(:));
specificTestImg = specificTestImg - imgmean;  % 中心化
specificTestImgObj = W' * specificTestImg;

% 找到最近的训练图片
distance = 1000000000000;
for k = 1:col_of_data
    temp = norm(specificTestImgObj - reference(:, k));
    if distance > temp
        aimone = k;
        distance = temp;
    end
end
trainImg = trainImageData(:, aimone);
trainImg = reshape(trainImg, [128, 128]);

% 展示特定测试图片及其最接近的训练图片
figure;
subplot(1, 2, 1);
imshow(reshape(specificTestImg, [128, 128]), []);
title('特定测试图片');

subplot(1, 2, 2);
imshow(trainImg, []);
title('最接近的训练图片');

% 选择特定图片进行测试并展示结果
specificTestImgPath = 'E:\majority\face_recognition\lzy3.jpg';
specificTestImg = imread(specificTestImgPath);
if size(specificTestImg, 3) == 3  % 检查是否为RGB图像
    specificTestImg = rgb2gray(specificTestImg);  % 转换为灰度图
end
specificTestImg = imresize(specificTestImg, [128 128]);  % 调整大小为128x128
specificTestImg = double(specificTestImg(:));
specificTestImg = specificTestImg - imgmean;  % 中心化
specificTestImgObj = W' * specificTestImg;

% 找到最近的训练图片
distance = 1000000000000;
for k = 1:col_of_data
    temp = norm(specificTestImgObj - reference(:, k));
    if distance > temp
        aimone = k;
        distance = temp;
    end
end
trainImg = trainImageData(:, aimone);
trainImg = reshape(trainImg, [128, 128]);

% 展示特定测试图片及其最接近的训练图片
figure;
subplot(1, 2, 1);
imshow(reshape(specificTestImg, [128, 128]), []);
title('特定测试图片');

subplot(1, 2, 2);
imshow(trainImg, []);
title('最接近的训练图片');



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