%% Users parameters
FOVx = 5000;       %[um]
FOVy = 5000;       %[um]
W = 5000;          %[um] - transducer width
D = 100;          %[um] - vessel diameter
F = 10;            %[MHz]
Z = 1000;          %[um]
C = 1e-4;          %[#bubbles/um^3]
Csound = 1540*1e6; %[um/sec]
psf_resolution = 500; 
Ncycles = 1;      
iter_num = 50; 

%% Calculated parameters
lamda = Csound/(F*1e6);  %[um]
FWHM = 0.886*lamda*Z/W;  %[um]
sigma = 2.355*FWHM;      %[um]
psf = fspecial('gaussian', psf_resolution, sigma);
psf = psf./max(psf(:));
up_lim = Z-floor(D/2);
down_lim = Z+ceil(D/2);
bubbles_num = C*FOVx*D;
X_population = 1:FOVx;
Y_population = up_lim:down_lim;
R_blur = ceil(FWHM);

%% Image declaration 
image = zeros(FOVx, FOVy);
true_image = zeros(FOVx, FOVy);
true_image(up_lim:down_lim,:) = 1;

%% Simulation
for t=1:iter_num
    x = randsample(X_population,bubbles_num);
    y = randsample(Y_population,bubbles_num);
    mask = zeros(FOVx, FOVy);
    for i = 1:bubbles_num
        x_i = x(i);
        y_i = y(i);
        lim_x0 = max(1,x_i-R_blur);
        lim_xf = min(FOVx,x_i+R_blur);
        lim_y0 = max(1,y_i-R_blur);
        lim_yf = min(FOVy,y_i+R_blur);
        if sum(sum(mask(lim_y0:lim_yf, lim_x0:lim_xf))) == 0
            mask(y_i,x_i) = 1;
        end
    end
    mask = conv2(mask,psf,'same'); 
    peaks = imregionalmax(mask);  
    image = image + peaks;
    display(t)
end
imshow(image)