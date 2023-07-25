
% im1 = 202;
% im2 = 203;
% z = [46.18, -450];  %% perfect match

% im1 = 194;
% im2 = 197;
% z = [420, -6];   %% perfect match 2
close all
clear, clc

rmax = 50;         % max distance (esclude coordinales where lines are too distant)
tr_coef  = 0.1;     % max sen gamma (esclude "less paralell" lines)
max_dist = 1000;    % max dist -->> 0 in normalization  ????? FUCTION OF step ???

ang = 45;           % rotation stem in gradi
step = 0.04;        % translation step in fraction [-1,1]; 0.1=21steps, 0.04=51steps

m = [];
z_rad = 1000;         % extrem of full grid ('radius' from 0 in px)
m(:,:,1) = meshgrid(-1:step:1)';   % 21 coordinates, includes 0,0
m(:,:,2) = meshgrid(-1:step:1);
z_id = m*z_rad;
grid_step = z_id(2,1,1)-z_id(1,1,1); % pixel difference

rot = 0:ang:360-ang;
p = [500, 500];  %% center of the image -> 00 for piece A

%% Input image 195 - 198; 200 - 203
% im1 = 202;
% im2 = 197;

pieces = [194:198, 200:203];
All_cost = zeros(size(m,2),size(m,2),size(rot,2),9,9);
Nnorm_cost = All_cost;

All_cost2=All_cost;

for f1=1:9          % - select fixed fragment
    % read image
    im1=pieces(f1);
    [alfa1, R1, I1] = read_info(im1);
    
    for f2=1:9      % - select moving and rotating fragment
        % read image
        im2=pieces(f2);
        [alfa2, R2, I2] = read_info(im2);
        
        R_cost = zeros(size(m,2),size(m,2),size(rot,2));
        if eq(f1,f2), R_norm = R_cost-1; R_norm2=R_norm;
        else            
            % initialization
            %R_cost = zeros(size(m,2),size(m,2),size(rot,2));            
            for t = 1:length(rot)
                % start iteration
                teta = -rot(t)*pi/180;  % rotation of F2
                
                for ix=1:size(z_id,1)
                    for iy=1:size(z_id,1)                        
                        z = z_id([ix,iy]); % z = squeeze(z_id(i,j,:))'; % ccordinate of F2
                        
                        n_linesf1 = size(alfa1,1);
                        n_linesf2 = size(alfa2,1);
                        
                        cost_matrix = zeros(n_linesf1,n_linesf2);                       
                        
                        for i=1:n_linesf1
                            for j=1:n_linesf2                                
                                
                                % translate reference point to the center
                                [beta1,R_new1, x1, y1, y_new1] = translation2(alfa1(i),R1(i),p);
                                [beta2,R_new2, x2, y2, y_new2] = translation2(alfa2(j),R2(j),p);
                                
                                [beta3,R_new3, x3, y3, y_new3] = translation2(beta2+teta,R_new2,-z); % shift and rot line 2
                                R_new4 = dist_punto_retta(beta1,R_new1, z);  % dist from new poit to line 1
                                
                                % Visualize lines
                                %  line_visualization(x1,x2,x3,y_new1,y_new2,y_new3,R_new1,R_new2,R_new3, z)
                                
                                %% distance between 2 lines
                                gamma = beta1-beta3;
                                coef = abs(sin(gamma));
                                
                                dist1 = (R_new1^2+R_new3^2-2*abs(R_new1*R_new3)*cos(gamma))^0.5;
                                dist2 = (R_new2^2+R_new4^2-2*abs(R_new2*R_new4)*cos(gamma))^0.5;
                                
                                if coef < tr_coef
                                    cost = (dist1+dist2);%/2; % cost = max(dist1,dist2);
                                else
                                    cost = max_dist;
                                end                                
                                cost_matrix(i,j) = cost;                                
                            end
                        end
                        [assignment,tot_cost] = munkres(cost_matrix); % LAP   
                        R_cost(iy,ix,t)=tot_cost;  
                        % option
                        r_c = min(sum(assignment.*cost_matrix));
                        R_cost2(iy,ix,t)=r_c;
                    end
                end
            end   
            
            R_norm = max(1-R_cost./rmax,0); % normalization
            R_norm2 = max(1-R_cost2./rmax,0); % normalization
        end
        Nnorm_cost(:,:,:,f2,f1) = R_cost;
        All_cost(:,:,:,f2,f1) = R_norm;
        All_cost2(:,:,:,f2,f1) = R_norm2;
    end
end

load('R_mask51_45cont2.mat', 'R_mask')
rneg = (R_mask<0)*(-1);
R_line = All_cost.*R_mask+rneg;
R_line2 = All_cost2.*R_mask+rneg;

% small correction of mask
for jj=1:9, R_line(:,:,:,jj,jj)=-1; end
R_line =R_line*2;
R_line(R_line<0) = -0.5;

for jj=1:9, R_line2(:,:,:,jj,jj)=-1; end
R_line2 = R_line2*2;
R_line2(R_line2<0) = -0.5;

save('R_line51_45_verLAP_fake2.mat', 'R_line2')
save('R_line51_45_verLAP_fakeNEW.mat', 'R_line') % full LAP

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visualize Rij matrices

jj = 3; % rotation

% Visualize matrices of cost befor norm
figure;
t1 = tiledlayout(9,9);
t1.Padding = 'none';
t1.TileSpacing = 'none';
for f1=1:9
    for f2=1:9
        %jj = 5;
        nexttile
        C = Nnorm_cost(:,:,jj,f2,f1);
        image(C,'CDataMapping','scaled'); colorbar
    end
end


% Visualize matrices of normilized cost
figure;
t1 = tiledlayout(9,9);
t1.Padding = 'none';
t1.TileSpacing = 'none';
for f1=1:9
    for f2=1:9
        %jj = 5;
        nexttile
        C = All_cost(:,:,jj,f2,f1);
        image(C,'CDataMapping','scaled'); colorbar
    end
end

% Visualize Rij matrices
figure;
t1 = tiledlayout(9,9);
t1.Padding = 'none';
t1.TileSpacing = 'none';
for f1=1:9
    for f2=1:9
        %jj = 1;
        nexttile
        C = R_line(:,:,jj,f2,f1);
        image(C,'CDataMapping','scaled'); colorbar
    end
end
