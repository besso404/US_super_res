%% Users parameters
FOVx = 5000;                    %[um]
FOVy = 5000;                    %[um]
W = 5000;                       %[um] - transducer width
D = 15;                         %[um] - vessel diameter
F = 10;                         %[MHz]
Z = 1000;                       %[um]
Z0 = 10000;                     %[um]
Csound = 1540*1e6;              %[um/sec]
psf_resolution = 50; 
Ncycles = 1;      
sim_len = 5000; 
ppm = 0.1;                      %[pixel/um]
mu_u = 1040;                    %[um/sec]
std_u = 100;                    %[um/sec]     
std_v = 0.5;
u = normrnd(mu_u,std_u);          %[um/sec]
v = normrnd(0,std_v);              %[um/sec]
FR = 30;                       %[frame/sec]


%% Calculated parameters
lamda = Csound/(F*1e6);         %[um]
FWHM_lat = 0.886*ppm*lamda*Z0/W; %[pixel]
sigma_lat = FWHM_lat/2.355;     %[pixel]
FOVx_ = floor(FOVx*ppm);
FOVy_ = floor(FOVy*ppm);
FWHM_ax = Ncycles*lamda*ppm/2;  %[pixel]
sigma_ax = FWHM_ax/2.355;       %[pixel]
up_lim = floor((Z-(D/2))*ppm);
down_lim = ceil((Z+(D/2))*ppm);
dt = 1/FR;                       %[sec/frame]
sigma_y = (D/2)^0.5;             %[um^0.5]


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
true_image(up_lim:down_lim,:) = 1;
image = double(zeros(FOVy_, FOVx_));
sample_im = zeros(FOVy_, FOVx_);
recon_im = zeros(FOVy_, FOVx_);

% RGB
true_image = cat(3, true_image, 0.1*true_image, 0.1*true_image);
background = true_image;

%% Simulation

f = figure;
a1 = subplot(2,2,1);
h1 = imshow(true_image);
set(get(a1, 'title'), 'string', 'True Image of Blood Vessel - flow');

a2 = subplot(2,2,2);
h2 = imshow(sample);
set(get(a2, 'title'), 'string', 'Sampled Image of Blood Vessel');

a3 = subplot(2,2,3);
h3 = imshow(image);
set(get(a3, 'title'), 'string',...
                        'Reconstructed SuperRes Image of Blood Vessel');

%LK - optical flow
opticFlow = opticalFlowFarneback('NeighborhoodSize',7);
a4 = subplot(2,2,4);
set(gca, 'xtick', []);
set(gca, 'ytick', []);
set(get(a4, 'title'), 'string',...
                        'Farneback Optical Flow');
movegui(f);
hViewPanel = uipanel(f);
hViewPanel.Position = get(a4,'Position');
hPlot = axes(hViewPanel);




bubbles = [struct('y', normrnd(Z,(D/2)^0.5)*ppm, 'x', 1,...
    'u', normrnd(mu_u,std_u)*ppm*dt,...
    'v', normrnd(0,std_v)*ppm*dt  , 't0', 1)]; 

exitted_frame = 0;


boundsx = (size(psf,1)-1)/2;
boundsy = (size(psf,2)-1)/2;

% Collect U vectors
U = [];
for t = 1:sim_len
    
    mask = zeros(FOVx_, FOVy_);
    
    if (mod(t, 30) == 0)
        bubbles = [bubbles; struct('y', normrnd(Z,(D/2)^0.5)*ppm,...
            'x', 1, 'u', normrnd(mu_u,std_u)*ppm*dt,...
            'v', normrnd(0,std_v)*ppm*dt, 't0', t)]; 
    end
    
    for b=1:length(bubbles)
        
        bubbles(b).y = bubbles(b).y + (t-bubbles(b).t0)*bubbles(b).v;

        bubbles(b).x = 1 + (t-bubbles(b).t0)*bubbles(b).u;

        if (bubbles(b).y < up_lim + abs(bubbles(b).v) || bubbles(b).y > down_lim - abs(bubbles(b).v))
            [bubbles(b).u,bubbles(b).v] = elastic_collision(bubbles(b).u,bubbles(b).v);
        end
        
        if (bubbles(b).x > FOVx_ - bubbles(b).u)
            exitted_frame = b;
        end

        mask(round(bubbles(b).y), min(FOVx_,round(bubbles(b).x))) = 1;
    
    end
    
    image = background + cat(3,conv2(mask, psf, 'same'));
    sample_im = conv2(mask, psf, 'same');
    %noise
    sample_im = imnoise(sample_im,'gaussian');
    corr = xcorr2(sample_im,psf); 
    corr = corr(boundsx:size(corr,1)-boundsx-1,boundsy:size(corr,2)-boundsy-1);
    corr = log10(corr+eps(0));
    corr(corr<max(corr(:))*0.8) = 0;
    corr = imregionalmax(corr);
    
    sample = sample + mask;
    image = image + corr;
    recon_im = recon_im + corr;
    
    set(h1, 'CData', image);
    set(h2, 'CData', sample_im);
    set(h3, 'CData', recon_im);
    
    
    if (exitted_frame > 0)
        bubbles(exitted_frame,:) = [];
        exitted_frame = 0;
    end
    
    circ = strel('disk',5);
    dil = imdilate(corr,circ);
    
    %flow
    flow = estimateFlow(opticFlow,dil);
    vy = flow.Vy(flow.Vy~=0);
    vx = flow.Vx(flow.Magnitude > 1);
    
    v_ = sprintf('%.2f',mean(vy(:)));
    U_ = mean(vx(:));
    U = [U; U_];
    u_ = sprintf('%.2f',mean(vx(:)));
    
    title_str = join(["Iteration #", int2str(t)," u:" , u_,...
        " [pixel/frame], v:", v_, " [pixel/frame]"]);
    
    imshow(dil);
    hold on
    plot(flow,'DecimationFactor',[50 10],'ScaleFactor',10,'Parent',hPlot);
    hViewPanel.Title = title_str;  
    hold off
    drawnow;
end

figure;
histogram(U,50)