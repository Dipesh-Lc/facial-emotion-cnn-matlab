function detectEmotions(modelFile)
% detectEmotions - Load a trained emotion CNN, manually select face region,
% and classify emotions based on the selected face only.
%
% Usage:
%   detectEmotions()                        % uses 'emotionCNN10.mat' in current folder
%   detectEmotions('path/to/model.mat')
%
% Prompts the user to select image files, displays each image,
% allows manual face-region selection via mouse, classifies emotion,
% and shows the result.

    if nargin < 1 || isempty(modelFile)
        modelFile = 'emotionCNN10.mat';
    end

    % Load the trained network
    try
        data       = load(modelFile);
        emotionNet = data.emotionNet10;
    catch ME
        error('Could not load model from %s: %s', modelFile, ME.message);
    end

    % Determine network input size
    inputSize = emotionNet.Layers(1).InputSize(1:2);

    % Prompt user to select images
    [files, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files'}, ...
                              'Select Image Files', 'MultiSelect', 'on');
    if isequal(files,0)
        disp('No images selected. Exiting.');
        return;
    end
    if ischar(files)
        files = {files};
    end

    % Process each selected image
    for i = 1:numel(files)
        imgPath = fullfile(path, files{i});
        try
            I = imread(imgPath);
        catch
            warning('Could not read image: %s. Skipping.', imgPath);
            continue;
        end

        % Display image and prompt for face region
        fig = figure('Name', files{i}, 'NumberTitle', 'off');
        imshow(I);
        title('Draw a rectangle around the face, then double-click inside it to confirm', 'FontSize', 12);

        % Let user draw rectangle
        hRect = imrect;
        position = wait(hRect);  % [x y width height]
        close(fig);

        % Crop selected face region
        faceRegion = imcrop(I, position);
        if isempty(faceRegion)
            warning('No region selected for %s. Skipping.', files{i});
            continue;
        end

        % Handle grayscale
        if size(faceRegion,3) == 1
            faceRegion = repmat(faceRegion, [1 1 3]);
        end

        % Resize to network input size
        faceResized = imresize(faceRegion, inputSize);

        % Classify emotion
        [label, score] = classify(emotionNet, faceResized);
        pct = max(score) * 100;

        % Display face with label
        figure('Name', ['Result - ' files{i}], 'NumberTitle', 'off');
        imshow(faceRegion);
        title(sprintf('%s (%.1f%%)', string(label), pct), 'FontSize', 14, 'Color', 'w', 'BackgroundColor', 'k');
    end
end
