%% Users parameters
FOVx = 5000;       %[um]
FOVy = 5000;       %[um]
W = 5000;          %[um] - transducer width
D = 500;           %[um] - vessel diameter
F = 10;            %[MHz]
Z = 1000;          %[um]
Csound = 1540*1e6; %[um/sec]
psf_resolution = 15; 
Ncycles = 1;      
sim_len = 1000; 
ppm = 0.1;         %[pixel/um]

u = 10000;         %[um/sec]
v = 10;             %[um/sec]
FR = 1e3;          %[Hz]
epsilon = 0.1;     %[sec]

%% Calculated parameters

lamda = Csound/(F*1e6);  %[um]
FWHM = 0.886*ppm*lamda*Z/W;  %[um]
sigma = 2.355*FWHM;      %[um]
FOVx_ = floor(FOVx*ppm);
FOVy_ = floor(FOVy*ppm);
psf = fspecial('gaussian', psf_resolution, sigma);
psf = psf./max(psf(:));
up_lim = floor((Z-(D/2))*ppm);
down_lim = ceil((Z+(D/2))*ppm);
R_blur = ceil(FWHM);
dt = 1/FR;          %[sec]

%% Image declaration 
background = zeros(FOVy_, FOVx_);
sample = double(zeros(FOVy_, FOVx_));
true_image = zeros(FOVy_, FOVx_);
true_image(up_lim:down_lim,:) = 1;

% RGB
true_image = cat(3, true_image, 0.1*true_image, 0.1*true_image);
background = true_image;

%% Simulation

figure;
h = imshow(true_image);

bubbles = [struct('y', round(Z*ppm), 'x', 1, 'u', u, 'v', v, 't0', 1)]; 

exitted_frame = 0;

for t = 1:sim_len
    
    mask = zeros(FOVx_, FOVy_);
    
    if (mod(t, 100) == 0)
        bubbles = [bubbles; struct('y', round(Z*ppm), 'x', 1, 'u', u, 'v', v, 't0', t)]; 
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
    
    set(h, 'CData', image);
    drawnow;
    
    if (exitted_frame > 0)
        bubbles(exitted_frame,:) = [];
        exitted_frame = 0;
    end
    
end


