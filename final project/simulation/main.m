%% Users parameters
FOVx = 5000;       %[um]
FOVy = 5000;       %[um]
W = 5000;          %[um] - transducer width
D = 100;           %[um] - vessel diameter
F = 10;            %[MHz]
Z = 1000;          %[um]
C = 9e-4;          %[#bubbles/um^3]
Csound = 1540*1e6; %[um/sec]
psf_resolution = 30; 
Ncycles = 1;      
iter_num = 200; 
ppm = 0.1;         %[pixel/um]

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
bubbles_num = floor(C*FOVx_*D);
X_population = 1:FOVx_;
Y_population = up_lim:down_lim-1;
R_blur = ceil(FWHM);

%% Image declaration 
image = zeros(FOVy_, FOVx_);
sample = zeros(FOVy_, FOVx_);
true_image = zeros(FOVy_, FOVx_);
true_image(up_lim:down_lim,:) = 1;

%% Simulation
figure

subplot(1,3,1)
imagesc(true_image);
subplot(1,3,2)
imagesc(sample);
subplot(1,3,3)
imagesc(image);

for t=1:iter_num
    x = randsample(X_population,bubbles_num,true);
    y = randsample(Y_population,bubbles_num,true);
    mask = zeros(FOVx_, FOVy_);
    for i = 1:bubbles_num
        x_i = x(i);
        y_i = y(i);
        lim_x0 = max(1,x_i-R_blur);
        lim_xf = min(FOVx_,x_i+R_blur);
        lim_y0 = max(1,y_i-R_blur);
        lim_yf = min(FOVy_,y_i+R_blur);
        if sum(sum(mask(lim_y0:lim_yf, lim_x0:lim_xf))) == 0
            mask(y_i,x_i) = 1;
        end
    end
    mask = conv2(mask,psf,'same'); 
    peaks = imregionalmax(mask); 
%     circ = strel('disk', 3);
%     peaks = imdilate(peaks,circ);
    sample = sample + mask;
    image = image + peaks;
    display(t)
    subplot(1,3,2)
    imagesc(sample);
    subplot(1,3,3)
    imagesc(image);
end
