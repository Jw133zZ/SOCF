function score = cal_spscore(response,sz)
%CAL_SPSCORE 此处显示有关此函数的摘要
%   此处显示详细说明
response(response(:)<=0) = 0;
response = fftshift(response);
response(response<0) =0;
use_sz = size(response);
center = floor((use_sz + 1)/2) + mod(use_sz +1,2);
range = zeros(numel(sz), 2);
for j = 1:numel(sz)
    range(j,:) = [0, sz(j) - 1] - floor(sz(j) / 2);
end
range_h = (center(1)+ range(1,1)) : (center(1) + range(1,2));
range_w = (center(2)+ range(2,1)) : (center(2) + range(2,2));
target_region = zeros(use_sz);
target_region(range_h,range_w) = response(range_h,range_w);
score = sum(target_region(:))/sum(response(:));
if isnan(score)
    score = 0;
end
end

