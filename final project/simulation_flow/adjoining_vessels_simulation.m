clear all; close all; clc;
%% Users parameters
FOVx = 5000;                    %[um]
FOVy = 5000;                    %[um]
W = 5000;                       %[um] - transducer width
D = 15;                         %[um] - vessel diameter
F = 10;                         %[MHz]
Z1 = 1000;                       %[um]
X2 = 1500;                       %[um]
Z0 = 10000;                     %[um]
Csound = 1540*1e6;              %[um/sec]
psf_resolution = 50;
Ncycles = 1;      
sim_len = 3000; 
ppm = 0.1;                      %[pixel/um]
mu_u = 1040;                    %[um/sec]
std_u = 100;                    %[um/sec]     
std_v = 0.5;
u = normrnd(mu_u,std_u);        %[um/sec]
v = normrnd(0,std_v);           %[um/sec]
FR = 30;                        %[frame/sec]
to_record = false;

%% Calculated parameters
lamda = Csound/(F*1e6);         %[um]
FWHM_lat = 0.886*ppm*lamda*Z0/W;%[pixel]
sigma_lat = FWHM_lat/2.355;     %[pixel]
FOVx_ = floor(FOVx*ppm);
FOVy_ = floor(FOVy*ppm);
FWHM_ax = Ncycles*lamda*ppm/2;  %[pixel]
sigma_ax = FWHM_ax/2.355;       %[pixel]
up_lim1 = floor((Z1-(D/2))*ppm);
down_lim1 = ceil((Z1+(D/2))*ppm);
up_lim2 = floor((X2-(D/2))*ppm);
down_lim2 = ceil((X2+(D/2))*ppm);
dt = 1/FR;                       %[sec/frame]
sigma_y = (D/2)^0.5;             %[um^0.5]
real_U = [];
real_V = [];

%% psf 
psf_lat = easygauss(floor(-psf_resolution/2)+1:floor(psf_resolution/2)...
    , 0, sigma_lat);
psf_ax = easygauss(floor(-psf_resolution/2)+1:floor(psf_resolution/2)...
    , 0, sigma_ax);
psf = conv2(psf_ax',psf_lat);

%% Image declaration 
background = zeros(FOVy_, FOVx_);

sample = double(zeros(FOVy_, FOVx_));

true_image = zeros(FOVy_, FOVx_);

true_image(up_lim1:down_lim1,:) = 1;
true_image(:,up_lim2:down_lim2) = 1;

image = double(zeros(FOVy_, FOVx_));
sample_im = zeros(FOVy_, FOVx_);
recon_im = zeros(FOVy_, FOVx_);
circ1 = strel('disk',2);
circ2 = strel('disk',5);

% RGB
true_image = cat(3, true_image, 0.1*true_image, 0.1*true_image);
background = true_image;

% 1mm scale bar implementation
mm = round(1000*ppm);                % 1 mm in [pixel]
bar_mask = zeros(FOVy_, FOVx_);
bar_mask(round(0.9*size(bar_mask,1)):round(0.9*size(bar_mask,1))+1,...
    round(0.1*size(bar_mask,2))+1:round(0.1*size(bar_mask,2))+mm) = ...
    ones(2,mm);

%% Display / Setup

f = figure;
a1 = subplot(2,2,1);
h1 = imshow(true_image);
set(get(a1, 'title'), 'string', 'True Image of Blood Vessels - flow');

a2 = subplot(2,2,2);
h2 = imshow(sample);
set(get(a2, 'title'), 'string', 'Sampled Image of Blood Vessels');

a3 = subplot(2,2,3);
h3 = imshow(image);
set(get(a3, 'title'), 'string',...
                        'Reconstructed SuperRes Image of Blood Vessels');

%LK - optical flow
opticFlow = opticalFlowFarneback('NeighborhoodSize', 7,'NumPyramidLevels', 3);
a4 = subplot(2,2,4);
set(gca, 'xtick', []);
set(gca, 'ytick', []);
set(get(a4, 'title'), 'string',...
                        'Farneback Optical Flow');
movegui(f);
hViewPanel = uipanel(f);
hViewPanel.Position = get(a4,'Position');
hPlot = axes(hViewPanel);

%bubbles declaration
bubbles1 = [struct('y', normrnd(Z1,(D/2)^0.5)*ppm, 'x', 1, ...
    'u', normrnd(mu_u,std_u)*ppm*dt,...
    'v', normrnd(0,std_v)*ppm*dt  , 't0', 1)];

bubbles2 = [struct('y', 1, 'x', normrnd(X2,(D/2)^0.5)*ppm, ...
    'u', normrnd(0,std_v)*ppm*dt, ...
    'v', normrnd(mu_u,std_u)*ppm*dt , 't0', 1)];
real_U = [real_U; bubbles1.u bubbles2.u];
real_V = [real_V; bubbles1.v bubbles2.v];

exitted_frame1 = 0;
exitted_frame2 = 0;

boundsx = ceil((size(psf,1)-1)/2);
boundsy = ceil((size(psf,2)-1)/2);

% Collect U and V sampling
U = [];
V = [];

% Videowriter
if to_record
    vid_title = join(['Flow simulation - FR - ', int2str(FR), ...
            ', number of iterations - ' , int2str(sim_len), '.avi']);
    vid = VideoWriter(vid_title, 'Motion JPEG AVI');
    vid.FrameRate = 30;
    open(vid);
end

ann = annotation('textbox', [0.47 0.5 0.1 0.1],'String',...
   num2str(0,'Test text %d'),'EdgeColor', 'none','HorizontalAlignment', 'center');

%% Simulation
for t = 1:sim_len
    
    set(ann, 'string', join(["Iteration #", int2str(t)]));
    mask = zeros(FOVx_, FOVy_);
    
    if (mod(t, 30) == 0)
        bubbles1 = [bubbles1; struct('y', normrnd(Z1,(D/2)^0.5)*ppm, 'x', 1, ...
            'u', normrnd(mu_u,std_u)*ppm*dt,...
            'v', normrnd(0,std_v)*ppm*dt  , 't0', t)];

        bubbles2 = [bubbles2; struct('y', 1, 'x', normrnd(X2,(D/2)^0.5)*ppm, ...
            'u', normrnd(0,std_v)*ppm*dt, ...
            'v', normrnd(mu_u,std_u)*ppm*dt , 't0', t)];
        real_U = [real_U; bubbles1(end).u bubbles2(end).u];
        real_V = [real_V; bubbles1(end).v bubbles2(end).v];
    end
    
    for b=1:length(bubbles1)
        
        bubbles1(b).y = bubbles1(b).y + (t-bubbles1(b).t0)*bubbles1(b).v;

        bubbles1(b).x = 1 + (t-bubbles1(b).t0)*bubbles1(b).u;

        if (bubbles1(b).y < up_lim1 + abs(bubbles1(b).v) || bubbles1(b).y > down_lim1 - abs(bubbles1(b).v))
            [bubbles1(b).u,bubbles1(b).v] = elastic_collision(bubbles1(b).u,bubbles1(b).v);
        end
        
        if (bubbles1(b).x > FOVx_ - bubbles1(b).u)
            exitted_frame1 = b;
        end

        mask(ceil(bubbles1(b).y), min(FOVx_,round(bubbles1(b).x))) = 1;
    
    end
    
    for b=1:length(bubbles2)
        
        bubbles2(b).y = 1 + (t-bubbles2(b).t0)*bubbles2(b).v;

        bubbles2(b).x = bubbles2(b).x + (t-bubbles2(b).t0)*bubbles2(b).u;
        
        if (bubbles2(b).x < down_lim2 + abs(bubbles2(b).u) || bubbles2(b).x > up_lim2 - abs(bubbles2(b).u))
            [bubbles2(b).v, bubbles2(b).u] = elastic_collision(bubbles2(b).v, bubbles2(b).u);
        end
             
        if (bubbles2(b).y > FOVy_ - bubbles2(b).v)
            exitted_frame2 = b;
        end

        mask(min(FOVy_,round(bubbles2(b).y)), ceil(bubbles2(b).x)) = 1;
    
    end
    
    image = background + cat(3,conv2(mask, psf, 'same'));
    sample_im = conv2(mask, psf, 'same');
    
    %noise
    sample_im = imnoise(sample_im,'gaussian');
    corr = xcorr2(sample_im,psf); 
    corr = corr(boundsx:size(corr,1)-boundsx,boundsy:size(corr,2)-boundsy);
    corr = log10(corr+eps(0));
    corr(corr<max(corr(:))*0.8) = 0;
    corr = imregionalmax(corr);
    
    sample = sample + mask;
    image = background + imdilate(mask,circ1);
    recon_im = recon_im + corr;
    
    set(h1, 'CData', image+bar_mask);
    set(h2, 'CData', sample_im+bar_mask);
    set(h3, 'CData', recon_im+bar_mask);
    
    
    if (exitted_frame1 > 0)
        bubbles1(exitted_frame1,:) = [];
        exitted_frame1 = 0;
    end
    
    if (exitted_frame2 > 0)
        bubbles2(exitted_frame2,:) = [];
        exitted_frame2 = 0;
    end
    
    dil = imdilate(corr,circ2);
    
    %flow
    flow = estimateFlow(opticFlow,dil);
    vy = flow.Vy(flow.Vy > 1);
    vx = flow.Vx(flow.Vx > 1);
    
    V_= mean(vy(:));
    V = [V; V_];
    v_ = sprintf('%.2f',mean(vy(:)));
    U_ = mean(vx(:));
    U = [U; U_];
    u_ = sprintf('%.2f',mean(vx(:)));
    
    title_str = join([" u:" , u_, ", v:", v_, " [pixel/frame]"]);
                
    imshow(dil+bar_mask);
    hold on
    plot(flow,'DecimationFactor',[50 10],'ScaleFactor',10,'Parent',hPlot);
    hViewPanel.Title = title_str;  
    hold off
    drawnow;
    
    if to_record 
        frame = getframe(gcf);
        writeVideo(vid,frame);
    end
end

if to_record
    close(vid);
end

figure;
subplot(2,1,1)
histogram(real_U(:,1),10,'Normalization','probability')
title('X Axis Velocity Histogram of Upper Vessel - Real');
xlabel('X Axis Velocity [pixel/frame]');
ylabel('Normalized Probability');
subplot(2,1,2)
histogram(U(U>min(real_U(:,1)) & U<max(real_U(:,1))),10,...
    'Normalization','probability')
title('X Axis Velocity Histogram of Upper Vessel - Sampled');
xlabel('X Axis Velocity [pixel/frame]');
ylabel('Normalized Probability');

figure;
subplot(2,1,1)
histogram(real_V(:),10,'Normalization','probability')
title('Y Axis Velocity Histogram - Real');
xlabel('Y Axis Velocity [pixel/frame]');
ylabel('Normalized Probability');
subplot(2,1,2)
histogram(V(V>min(real_V(:)) & V<max(real_V(:))),10,'Normalization','probability')
title('Y Axis Velocity Histogram - Sampled');
xlabel('Y Axis Velocity [pixel/frame]');
ylabel('Normalized Probability');

[mean(real_V(:)) mean(V)]
[mean(real_U(:,1)) mean(U(~isnan(U)))]
