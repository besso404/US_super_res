%% Users parameters
FOVx = 5000;                    %[um]
FOVy = 5000;                    %[um]
W = 5000;                       %[um] - transducer width
D = 500;                        %[um] - vessel diameter
F = 10;                         %[MHz]
Z = 1000;                       %[um]
Csound = 1540*1e6;              %[um/sec]
psf_resolution = 15; 
Ncycles = 1;      
sim_len = 5000; 
ppm = 0.1;                      %[pixel/um]

mu_u = 33333;                   %[um/sec]
u = normrnd(mu_u,3205);        %[um/sec]
v = normrnd(0,20);              %[um/sec]
FR = 300;                       %[frame/sec]
epsilon = 0.1;                  %[sec]

%% Calculated parameters

lamda = Csound/(F*1e6);         %[um]
FWHM = 0.886*ppm*lamda*Z/W;     %[um]
sigma = 2.355*FWHM;             %[um]
FOVx_ = floor(FOVx*ppm);
FOVy_ = floor(FOVy*ppm);
psf = fspecial('gaussian', psf_resolution, sigma);
psf = psf./max(psf(:));
up_lim = floor((Z-(D/2))*ppm);
down_lim = ceil((Z+(D/2))*ppm);
R_blur = ceil(FWHM);
dt = 1/FR;                       %[sec/frame]
sigma_y = (D/2)^0.5;             %[um^0.5]

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

figure
a1 = subplot(1,3,1);
h1 = imshow(true_image);
set(get(a1, 'title'), 'string', 'True Image of Blood Vessel - flow');

a2 = subplot(1,3,2);
h2 = imshow(sample);
set(get(a2, 'title'), 'string', 'Sampled Image of Blood Vessel');

a3 = subplot(1,3,3);
h3 = imshow(image);
set(get(a3, 'title'), 'string',...
                        'Reconstructed SuperRes Image of Blood Vessel');

bubbles = [struct('y', unifrnd(Z-D/2+1,Z+D/2)*ppm, 'x', 1,...
    'u', normrnd(mu_u,3205)*ppm*dt,...
    'v', normrnd(0,20)*ppm*dt  , 't0', 1)]; 

exitted_frame = 0;

LK - optical flow
opticFlow = opticalFlowFarneback();
h4 = figure;
movegui(h4);
hViewPanel = uipanel(h4);
hViewPanel.Title = 'LK optical flow';
hViewPanel.Position = [0 0 1 1];
hPlot = axes(hViewPanel);


boundsx = (size(psf,1)-1)/2;
boundsy = (size(psf,2)-1)/2;

% Collect U vectors
U = [];
for t = 1:sim_len
    
    mask = zeros(FOVx_, FOVy_);
    
    if (mod(t, 20) == 0)
        bubbles = [bubbles; struct('y', unifrnd(Z-D/2+1,Z+D/2)*ppm,...
            'x', 1, 'u', normrnd(mu_u,3205)*ppm*dt,...
            'v', normrnd(0,20)*ppm*dt, 't0', t)]; 
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

        mask(round(bubbles(b).y), round(bubbles(b).x)) = 1;
    
    end
    
    image = background + cat(3,conv2(mask, psf, 'same'));
    sample_im = conv2(mask, psf, 'same');
    
    corr = xcorr2(mask,psf); 
    corr = corr(boundsx:size(corr,1)-boundsx-1,boundsy:size(corr,2)-boundsy-1);
    corr = imregionalmax(corr);
    
    sample = sample + mask;
    image = image + corr;
    recon_im = recon_im + corr;
    
    set(h1, 'CData', image);
    set(h2, 'CData', sample_im);
    set(h3, 'CData', recon_im);
    drawnow;
    
    if (exitted_frame > 0)
        bubbles(exitted_frame,:) = [];
        exitted_frame = 0;
    end
    
    flow = estimateFlow(opticFlow,sample_im);
    vy = flow.Vy(flow.Vy~=0);
    vx = flow.Vx(flow.Magnitude > 0.75);
    
    v_ = sprintf('%.2f',mean(vy(:)));
    U_ = mean(vx(:));
    U = [U; U_];
    u_ = sprintf('%.2f',mean(vx(:)));
    title_str = join(["Iteration #", int2str(t)," u:" , u_, " v:",v_]);
    imshow(sample_im)
    hold on
    plot(flow,'DecimationFactor',[15 15],'ScaleFactor',10,'Parent',hPlot);
    hViewPanel.Title = title_str;
    hold off
end

figure;
histogram(U);
