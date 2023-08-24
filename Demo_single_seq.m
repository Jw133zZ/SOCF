
%   This script runs the implementation of IBRI for visual tracking.
%   Some functions are borrowed from other papers (SRDCF, UAV123 Benchmark)
%   and their copyright belongs to the paper's authors.

clear; clc; close all;
addpath(genpath('.'));

pathAnno = 'E:\UAV123_10fps\anno_10fps\UAV123_10fps\';
seqs = configSeqs_demo;

idx = 1;

for idxSeq=1:length(seqs)
    s = seqs{idxSeq};
    
    video_path = s.path;
        
    s.len = s.endFrame - s.startFrame + 1;
    s.s_frames = cell(s.len,1);
    nz	= strcat('%0',num2str(s.nz),'d'); %number of zeros in the name of image
    for i=1:s.len
        image_no = s.startFrame + (i-1);
        id = sprintf(nz,image_no);
        s.s_frames{i} = strcat(s.path,id,'.',s.ext);
    end
    
    img = imread(s.s_frames{1});
    [imgH,imgW,ch]=size(img);
    
    rect_anno = dlmread([pathAnno s.name '.txt']);
    numSeg = 20;
    
    [subSeqs, subAnno]=splitSeqTRE(s,numSeg,rect_anno);

    subS = subSeqs{1};
    subSeqs=[];
    subSeqs{1} = subS;

    subA = subAnno{1};
    subAnno=[];
    subAnno{1} = subA;
    results = [];

    subS = subSeqs{idx};

    subS.name = [subS.name '_' num2str(idx)];
    
    res = run_Ours(subS, 0, 0);

    res.len = subS.len;
    res.annoBegin = subS.annoBegin;
    res.startFrame = subS.startFrame;

    results{idx} = res;
    
    %   Load video information
    ground_truth = subAnno{1};

    gt_boxes = [ground_truth(:,1:2), ground_truth(:,1:2) + ground_truth(:,3:4) - ones(size(ground_truth,1), 2)];
   
pd_boxes = results{idx}.res;
thresholdSetOverlap = 0: 0.05 : 1;
success_num_overlap = zeros(1, numel(thresholdSetOverlap));
res = calcRectInt(rect_anno, pd_boxes);
for t = 1: length(thresholdSetOverlap)
    success_num_overlap(1, t) = sum(res > thresholdSetOverlap(t));
end
cur_AUC = mean(success_num_overlap) / size(gt_boxes, 1);
FPS_vid = results{idx}.fps;
display([seqs{idxSeq}.name  '---->' '   FPS:   ' num2str(FPS_vid)   '    op:   '   num2str(cur_AUC)]);
    

end