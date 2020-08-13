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
sim_len = 1500; 
ppm = 0.1;                      %[pixel/um]

u = normrnd(33333,3205);        %[um/sec]
v = normrnd(0,10);              %[um/sec]
FR = 300;                       %[Hz]
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
dt = 1/FR;                       %[sec]

%% Image declaration 
background = zeros(FOVy_, FOVx_);
sample = double(zeros(FOVy_, FOVx_));
true_image = zeros(FOVy_, FOVx_);
true_image(up_lim:down_lim,:) = 1;
image = double(zeros(FOVy_, FOVx_));
sample_im = zeros(FOVy_, FOVx_);
recun_im = zeros(FOVy_, FOVx_);

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

bubbles = [struct('y', floor(unifrnd(Z-D/2+1,Z+D/2)*ppm), 'x', 1,...
    'u', floor(normrnd(33333,3205)*ppm),...
    'v', floor(normrnd(0,10)*ppm)  , 't0', 1)]; 

exitted_frame = 0;

for t = 1:sim_len
    
    mask = zeros(FOVx_, FOVy_);
    
    if (mod(t, 100) == 0)
        bubbles = [bubbles; struct('y', floor(unifrnd(Z-D/2+1,Z+D/2)*ppm),...
            'x', 1, 'u', floor(normrnd(33333,3205)*ppm),...
            'v', floor(normrnd(0,10)*ppm), 't0', t)]; 
    end
    
    for b=1:length(bubbles)
        
        bubbles(b).y = bubbles(b).y + (t-bubbles(b).t0)*dt*bubbles(b).v*ppm;

        bubbles(b).x = 1 + (t-bubbles(b).t0)*dt*bubbles(b).u*ppm;

        if (bubbles(b).y < up_lim + abs(v)*ppm || bubbles(b).y > down_lim - abs(v)*ppm)
            [bubbles(b).u,bubbles(b).v] = elastic_collision(bubbles(b).u,bubbles(b).v);
        end
        
        if (bubbles(b).x > FOVx_ - dt*bubbles(b).u*ppm)
            exitted_frame = b;
        end

        mask(round(bubbles(b).y), round(bubbles(b).x)) = 1;
    
    end
    
    image = background + cat(3,conv2(mask, psf, 'same'));
    sample_im = conv2(mask, psf, 'same');
    peaks = imregionalmax(sample_im); 
    recun_im = recun_im + peaks;
    
    set(h1, 'CData', image);
    set(h2, 'CData', sample_im);
    set(h3, 'CData', recun_im);
    drawnow;
    
    if (exitted_frame > 0)
        bubbles(exitted_frame,:) = [];
        exitted_frame = 0;
    end
    
end

