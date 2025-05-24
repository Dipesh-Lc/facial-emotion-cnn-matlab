function detectEmotionsOnFacesMtcnn(modelFile)
% detectEmotionsOnFaces - Detect faces in full images with MTCNN, then classify emotions per face

% Usage:
%   detectEmotionsOnFaces()                  % uses 'emotionCNN10.mat'
%   detectEmotionsOnFaces('path/to/model.mat')

% Prompts user to select image files, then for each:
%   - Detects faces using MTCNN
%   - Crops each face, resizes, and classifies emotion
%   - Annotates original image with boxes and predicted emotions

    if nargin < 1 || isempty(modelFile)
        modelFile = 'emotionCNN10.mat';
    end

    % Load trained network
    try
        S = load(modelFile);
        net = S.emotionNet10;
    catch ME
        error('Could not load model from %s: %s', modelFile, ME.message);
    end

    % Create MTCNN detector
    % (make sure mtcnn.FaceDetection is on your path)
    detector = mtcnn.Detector();

    % Determine network input size
    inputSize = net.Layers(1).InputSize(1:2);

    % Prompt user for image files
    [files, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*','Image Files'}, ...
                              'Select Images', 'MultiSelect', 'on');
    if isequal(files,0)
        disp('No images selected. Exiting.');
        return;
    end
    if ischar(files); files = {files}; end

    % Process each image
    for i = 1:numel(files)
        imgPath = fullfile(path, files{i});
        I = imread(imgPath);

        % Detect faces with MTCNN
        [bboxes,scores,~] = detector.detect(I);
        if isempty(bboxes)
            warning('No faces detected in %s. Skipping.', files{i});
            continue;
        end

        labels = strings(size(bboxes,1),1);
        confidences = zeros(size(bboxes,1),1);

        % For each detected face, classify emotion
        for j = 1:size(bboxes,1)
            bb = bboxes(j,:);
            face = imcrop(I, bb);
            if isempty(face)
                labels(j) = '';
                confidences(j) = NaN;
                continue;
            end
            % Convert to RGB if grayscale
            if size(face,3) == 1
                face = repmat(face,[1 1 3]);
            end
            % Resize
            faceResized = imresize(face, inputSize);
            % Predict
            [lbl, score] = classify(net, faceResized);
            labels(j) = string(lbl);
            confidences(j) = max(score);
        end

        % Prepare annotations: 'Emotion (XX%)'
        ann = arrayfun(@(k) sprintf('%s (%.1f%%)', labels(k), confidences(k)*100), ...
                       1:numel(labels), 'UniformOutput', false);

        % Annotate image
        Iann = insertObjectAnnotation(I, 'rectangle', bboxes, ann, ...
                                      'FontSize',30, 'TextBoxOpacity',1, ...
                                      'LineWidth',7);

        % Display
        figure('Name', files{i});
        imshow(Iann);
        title('Detected Faces with Emotion Predictions','FontSize',15);
    end
end
