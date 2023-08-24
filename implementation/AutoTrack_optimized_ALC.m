% This function implements the ASRCF tracker.

function [results] = AutoTrack_optimized_ALC(params)
eta = params.eta;
rho = params.rho;
mu2 = params.mu2;
%   Setting parameters for local use.
admm_iterations = params.admm_iterations;
search_area_scale   = params.search_area_scale;
max_image_sample_size=params.max_image_sample_size;
min_image_sample_size=params.min_image_sample_size;
output_sigma_factor = params.output_sigma_factor;
% Scale parameters
num_scales=params.num_scales;
scale_sigma_factor=params.scale_sigma_factor;
scale_step=params.scale_step;
scale_lambda=params.scale_lambda;
scale_model_factor=params.scale_model_factor;
scale_model_max_area =params.scale_model_max_area;
lambda=params.admm_lambda;
features    = params.t_features;
video_path  = params.video_path;
s_frames    = params.s_frames;
pos         = floor(params.init_pos);
target_sz   = floor(params.wsize);
visualization  = params.visualization;
num_frames     = params.no_fram;
zeta=params.zeta;
newton_iterations = params.newton_iterations;
featureRatio = params.t_global.cell_size;
search_area = prod(target_sz * search_area_scale);
global_feat_params = params.t_global;
if search_area > max_image_sample_size
    currentScaleFactor = sqrt(search_area / max_image_sample_size);
elseif search_area <min_image_sample_size
    currentScaleFactor = sqrt(search_area / min_image_sample_size);
else
    currentScaleFactor = 1.0;
end
% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;
reg_sz= floor(base_target_sz/featureRatio);
% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        sz = floor(base_target_sz * search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        sz = repmat(sqrt(prod(base_target_sz * search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        sz = base_target_sz + sqrt(prod(base_target_sz * search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    otherwise
        error('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
end

% set the size to exactly match the cell size
sz = round(sz / featureRatio) * featureRatio;
use_sz = floor(sz/featureRatio);
filter_sz = use_sz;

% construct the label function- correlation output, 2D gaussian function,
% with a peak located upon the target

output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;
rg           = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
cg           = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);
[rs, cs]     = ndgrid(rg,cg);
y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf           = fft2(y); %   FFT of y.\

interp_sz = use_sz;

% construct cosine window
cos_window = single(hann(use_sz(1)+2)*hann(use_sz(2)+2)');
cos_window = cos_window(2:end-1,2:end-1);
try
    im = imread([video_path '/img/' s_frames{1}]);
catch
    try
        im = imread(s_frames{1});
    catch
        im = imread([video_path '/' s_frames{1}]);
    end
end
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        colorImage = false;
    else
        colorImage = true;
    end
else
    colorImage = false;
end

% compute feature dimensionality
feature_dim = 0;
for n = 1:length(features)
    
    if ~isfield(features{n}.fparams,'useForColor')
        features{n}.fparams.useForColor = true;
    end
    
    if ~isfield(features{n}.fparams,'useForGray')
        features{n}.fparams.useForGray = true;
    end
    
    if (features{n}.fparams.useForColor && colorImage) || (features{n}.fparams.useForGray && ~colorImage)
        feature_dim = feature_dim + features{n}.fparams.nDim;
    end
end

if size(im,3) > 1 && colorImage == false
    im = im(:,:,1);
end

%% SCALE ADAPTATION INITIALIZATION
% Use the translation filter to estimate the scale
scale_sigma = sqrt(num_scales) * scale_sigma_factor;
ss = (1:num_scales) - ceil(num_scales/2);
ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
ysf = single(fft(ys));
if mod(num_scales,2) == 0
    scale_window = single(hann(num_scales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(num_scales));
end
ss = 1:num_scales;
scaleFactors = scale_step.^(ceil(num_scales/2) - ss);
if scale_model_factor^2 * prod(target_sz) > scale_model_max_area
    scale_model_factor = sqrt(scale_model_max_area/prod(target_sz));
end
if prod(target_sz) >scale_model_max_area
    params.scale_model_factor = sqrt(scale_model_max_area/prod(target_sz));
end
scale_model_sz = floor(target_sz * scale_model_factor);

% set maximum and minimum scales
min_scale_factor = scale_step ^ ceil(log(max(5 ./sz)) / log(scale_step));
max_scale_factor =scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));

% Pre-computes the grid that is used for score optimization
ky = circshift(-floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2), [1, -floor((use_sz(1) - 1)/2)]);
kx = circshift(-floor((use_sz(2) - 1)/2) : ceil((use_sz(2) - 1)/2), [1, -floor((use_sz(2) - 1)/2)])';

% initialize the projection matrix (x,y,h,w)
rect_position = zeros(num_frames, 4);
time = 0;
loop_frame = 1;
A = ones([1,1,42]);
score_psr = zeros([1,1,42]);
for frame = 1:num_frames
    %load image
    try
        im = imread([video_path '/img/' s_frames{frame}]);
    catch
        try
            im = imread([s_frames{frame}]);
        catch
            im = imread([video_path '/' s_frames{frame}]);
        end
    end
    if size(im,3) > 1 && colorImage == false
        im = im(:,:,1);
    end
    tic();
    %% main loop
    
    if frame > 1
        pixel_template=get_pixels(im, pos, round(sz*currentScaleFactor), sz);
        xt=get_features(pixel_template,features,global_feat_params);
        xtf=fft2(bsxfun(@times,xt,cos_window));
        responsef=permute(sum(bsxfun(@times, conj(g_f), xtf), 3), [1 2 4 3]);
        % if we undersampled features, we want to interpolate the
        % response so it has the same size as the image patch
        responsef_padded = resizeDFT2(responsef, interp_sz);
        % response in the spatial domain
        response = ifft2(responsef_padded, 'symmetric');
        % find maximum peak
        [disp_row, disp_col] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, use_sz);
        % calculate translation
        translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor);
        %update position
        pos = pos + translation_vec;
    
        %%Scale Search

        xs = crop_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
        xsf = fft(xs,[],2);
        scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den+scale_lambda)));
        % find the maximum scale response
        recovered_scale = find(scale_response == max(scale_response(:)), 1);
        % update the scale
        currentScaleFactor = currentScaleFactor * scaleFactors(recovered_scale);
        if currentScaleFactor < min_scale_factor
            currentScaleFactor = min_scale_factor;
        elseif currentScaleFactor > max_scale_factor
            currentScaleFactor = max_scale_factor;
        end
    end
    target_sz =round(base_target_sz * currentScaleFactor);
    
    %save position
    rect_position(loop_frame,:) =[pos([2,1]) - (target_sz([2,1]))/2, target_sz([2,1])];
    
    if frame==1
        % extract training sample image region
        pixels = get_pixels(im,pos,round(sz*currentScaleFactor),sz);
        pixels = uint8(gather(pixels));
        x=get_features(pixels,features,global_feat_params);
        xf=fft2(bsxfun(@times,x,cos_window));
    else
        % use detection features
        shift_samp_pos = 2*pi * translation_vec ./(currentScaleFactor* sz);
        xf = shift_sample(xtf, shift_samp_pos, kx', ky');
    end
    %% 通道可靠性
    x_cf = real(ifft2(xf));
    for iii = 1:42
        score_psr(iii) = cal_psr(x_cf(:,:,iii));
%         score_psr(iii) = cal_apce(x_cf(:,:,iii));
    end
    beta_0 = A + eta.* (score_psr)./(mean(score_psr) - score_psr);
    beta_0 = max(0.96 , min(1.04,beta_0));
%     beta_0 = ones([1,1,42]);
    
    if  frame == 1
        [~,~,w]=init_regwindow(use_sz,reg_sz,params);
        g_pre= zeros(size(xf));
        mu = 0;
    else
        mu=zeta;
    end
    %% 显著性检测
    do_sa_target_size = 20;
    if min(target_sz) <= do_sa_target_size
        mask_filter = params.reg_window_min ./w;
    else
        Saliency_extract_sz = round(target_sz * 2);
        rgbpatch = get_pixels(im,pos,Saliency_extract_sz,Saliency_extract_sz);
        [origin_row , origin_col,~]=size(rgbpatch);
        do_sa_in_size = 32;
        rgbpatch = imresize(rgbpatch, [do_sa_in_size , do_sa_in_size]);
        %Spectral Residual光谱残差
        myFFT = fft2(rgb2gray(rgbpatch));
        myLogAmplitude = log(abs(myFFT));
        myPhase = angle(myFFT);
        mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', 3), 'replicate');
        saliencyMap = abs(ifft2(exp(mySpectralResidual + 1i * myPhase))).^2;
        %After Effect
        saliencyMap = mat2gray(imfilter(saliencyMap, fspecial('gaussian', [10, 10], 2.5))); %此为显著性区域图
        salpatch = imresize(saliencyMap, [ origin_row , origin_col ] ); %将显著性图放大到原始大小
        patchscale{1} = size(salpatch) ;
        window = cellfun(@(sz) hann(sz(1))*hann(sz(2))', patchscale, 'uniformoutput', false);
        salpatch = salpatch .* window{1} ;%抑制边缘
        sal_resize_scale=target_sz/currentScaleFactor/featureRatio * 2;
        salpatch_in_feature_size = imresize(salpatch , sal_resize_scale ); %缩小到滤波器大小
        
        m_in_feature_size = salpatch_in_feature_size;
        bi_thrs = params.bi_thrs ;
        m_in_feature_size(salpatch_in_feature_size(:) <= bi_thrs ) =  params.reg_window_max;
        m_in_feature_size(salpatch_in_feature_size(:) >  bi_thrs ) =  params.reg_window_min;      
        [m_mask_row,m_mask_col]=size(m_in_feature_size);
        m_input =  params.reg_window_max * ones(filter_sz);
        
        minput_rowmin = round((filter_sz(1)-m_mask_row)/2);
        minput_rowmax = minput_rowmin + m_mask_row - 1;
        
        minput_colmin = round((filter_sz(2)-m_mask_col)/2);
        minput_colmax = minput_colmin + m_mask_col - 1;   
        
        try
            m_input(minput_rowmin:minput_rowmax,minput_colmin:minput_colmax) = m_in_feature_size;
            
            reg_window_input_s2 =  params.reg_window_max *ones(filter_sz);
            
            reg_window_input_s2( m_input(:) <=1 & w(:) <=1 ) = params.reg_window_min;
        catch
            reg_window_input_s2 = w;
        end       
        try
            mask_filter = ones(size(reg_window_input_s2));
            mask_filter( reg_window_input_s2(:) >=1 ) = params.reg_window_min;
            mask_filter( reg_window_input_s2(:) <1 )  = params.reg_window_max;
            mask_filter = mask_filter ./ params.reg_window_max;
        catch
%             mask_filter = ones(size(reg_window_input_s2));
            mask_filter = params.reg_window_min ./w;
        end
   
 
    end
    
    g_f = single(zeros(size(xf)));
    h_f = g_f;
    l_f = h_f;
    gamma = 1;
    betha = 10;
    %      gamma_max = 10000;
    gamma_max = 0.1;
    if frame ==1
        beta_k = ones([1,1,42]);
    else
        beta_k = beta_0;
    end
    
    % ADMM solution
    T = prod(use_sz); 
    iter = 1;
    while (iter <= admm_iterations)
        if iter == 1
            phi = 0;
        else
            phi = mu2;
        end
        f_f0 = fft2(ifft2(g_f) .*  mask_filter);
        S_xbbx = sum(conj(xf) .* xf .* (beta_k.^2), 3);
%         S_xx = sum(conj(xf) .* xf, 3);
        S_betax = xf .* beta_k;
        Sg_pre= sum(conj(S_betax) .* g_pre, 3);
        Sgx_pre= bsxfun(@times, S_betax, Sg_pre);
        S_f0x = sum(conj(S_betax) .* f_f0, 3);
        S_xf0x = bsxfun(@times, S_betax, S_f0x);
        % subproblem g
        B = S_xbbx + T * (gamma + mu + phi);
        Shx_f = sum(conj(S_betax) .* h_f, 3);
        Slx_f = sum(conj(S_betax) .* l_f, 3);
        g_f = ((1/(T*(gamma + mu + phi)) * bsxfun(@times,  yf, S_betax)) - ((1/(gamma + mu + phi)) * l_f) +(gamma/(gamma + mu + phi)) * h_f)...
            + (mu/(gamma + mu + phi)) * g_pre  + (phi/(gamma + mu + phi)) * f_f0 - ...
            bsxfun(@rdivide,(1/(T*(gamma + mu + phi)) * bsxfun(@times, S_betax, (S_xbbx .*  yf)) + (mu/(gamma + mu + phi)) * Sgx_pre- ...
            (1/(gamma + mu + phi))* (bsxfun(@times, S_betax, Slx_f)) +(gamma/(gamma + mu + phi))* (bsxfun(@times, S_betax, Shx_f))...
             + (phi/(gamma + mu + phi))* S_xf0x), B);
        %   subproblem h
        lhd= T ./  (lambda*w .^2 + gamma*T);
        X=ifft2(gamma*(g_f + l_f));
        h=bsxfun(@times,lhd,X);
        h_f = fft2(h);
        
        % subproblem beta
        S_g_x = g_f .* xf;
        beta_k = (rho*T*beta_0 + sum(sum(bsxfun(@times,conj(S_g_x),yf),2),1))./...
            (sum(sum(bsxfun(@times,conj(S_g_x),S_g_x), 2),1) + T*rho);
        beta_k = max(0.9 , min(1.1,beta_k));
        %   update h
        l_f = l_f + (gamma * (g_f - h_f));
        
        %   update gamma
        gamma = min(betha* gamma, gamma_max);
        iter = iter+1;
    end
    
    
    % save the trained filters
    g_pre= g_f;

    %% Upadate Scale
    if frame==1
        xs = crop_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
    else
        xs= shift_sample_scale(im, pos, base_target_sz,xs,recovered_scale,currentScaleFactor*scaleFactors,scale_window,scale_model_sz);
    end
    
    xsf = fft(xs,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
    
    if frame == 1
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        sf_den = (1 - params.learning_rate_scale) * sf_den + params.learning_rate_scale * new_sf_den;
        sf_num = (1 - params.learning_rate_scale) * sf_num + params.learning_rate_scale * new_sf_num;
    end
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;
    time = time + toc();
    
    %%   visualization
    if visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if loop_frame == 1
            fig_handle = figure('Name', 'Tracking');
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(loop_frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            resp_sz = round(sz*currentScaleFactor*scaleFactors(recovered_scale));
            xs = floor(pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
            ys = floor(pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
            %sc_ind = floor((nScales - 1)/2) + 1;
            
            figure(fig_handle);
            imagesc(im_to_show);
            hold on;
            resp_handle = imagesc(xs, ys, fftshift(response)); colormap hsv;
            alpha(resp_handle, 0.2);
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(20, 30, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 16);
            text(20, 100, ['FPS : ' num2str(1/(time/loop_frame))], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 16);
            
            hold off;
        end
        drawnow
    end
    loop_frame = loop_frame + 1;
end

%   save resutls.
fps = loop_frame / time;
results.type = 'rect';
results.res = rect_position;
results.fps = fps;

end
