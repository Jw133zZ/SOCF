%   This function runs the AutoTrack tracker on the video specified in "seq".
%   This function borrowed from STRCF paper. 
%   details of some parameters are not presented in the paper, you can
%   refer to STRCF paper for more details.
function results = run_AutoTrack_ALC(seq,~,~)

%  Initialize path
addpath('feature/');
addpath('implementation/');
addpath('utils/');

%   HOG feature parameters
hog_params.cell_size = 4;
hog_params.nDim   = 31;

%   ColorName feature parameters
cn_params.nDim  =10;
cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;

%  Grayscale feature parameters
grayscale_params.nDim=1;
grayscale_params.colorspace='gray';
grayscale_params.cell_size = 4;

%   Global feature parameters 
params.t_features = {
    struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...  
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
  };

%  Global feature parameters
params.t_global.cell_size = 4;                  % Feature cell size

%   Search region + extended background parameters
params.search_area_shape = 'square';    % the shape of the training/detection window: 'proportional', 'square' or 'fix_padding'
params.search_area_scale =5;           % the size of the training/detection area proportional to the target size
params.min_image_sample_size = 150^2;   % Minimum area of image samples
params.max_image_sample_size = 200^2;   % Maximum area of image samples

%   Gaussian response parameter
params.output_sigma_factor =0.06;		% standard deviation of the desired correlation output (proportional to target)

%   Detection parameters
params.newton_iterations= 5;           % number of Newton's iteration to maximize the detection scores
% SAMF尺度
params.number_of_scales = 5;            % Number of scales to run the detector
params.scale_step = 1.01;   
%  Set files and gt
params.video_path = seq.path;
params.img_files = seq.s_frames;
params.wsize    = [seq.init_rect(1,4), seq.init_rect(1,3)];
params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);
params.s_frames = seq.s_frames;
params.no_fram  = seq.endFrame - seq.startFrame + 1;
params.seq_st_frame = seq.startFrame;
params.seq_en_frame = seq.endFrame;
params.ground_truth=seq.init_rect;

%   ADMM parameters, # of iteration, and lambda- mu and betha are set in
%   the main function.

params.admm_lambda =10000;

params.reg_window_max=1e5;
params.reg_window_min=1e-3;

%  Scale parameters
params.scale_sigma_factor=0.5;   
params.num_scales=33;
params.scale_step=1.03;
params.scale_model_factor = 1.0;
params.scale_model_max_area = 32*16;
params.hog_scale_cell_size = 4;  
params.scale_lambda = 1e-4;      
params.learning_rate_scale=0.025;
%occlusion params
params.tau=0.25;
params.M_learning_thre = 0.125;
params.repression_limit = 0.06;
params.penlty = 0.0430;
params.cita = 1;
%bucket
params.spatial = 11;
% channel reability
params.eta = 0.1;
params.rho = 0.0171;

% 显著性检测
params.bi_thrs = 0.15;
%   Debug and visualization
params.visualization = 1;
params.admm_iterations = 4; %4
%   目标感知滤波器参数
params.phi = 0.12;  %
params.zeta = 15;   %15
%   Run the main function
% results = AutoTrack_optimized(params);
results = AutoTrack_optimized_ALC_DR(params);  %干扰图正则化
% results = AutoTrack_optimized_ALC_AF(params);

