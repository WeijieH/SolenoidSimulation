%% Set parameters
clear
%Number of current loops
NumberOfTruns=100;
%radius of solenoid, in mm
sradius=35;
%length of solenoid, in mm
slength=150;
%Current, in A
scurrent=2;
% [half height, width] of simulation grid, in mm
ssize=[120, 80];
%Resolution, dot per mm
resolution=5;
%gap between first current loop and simulation grid in z direction, unit in mm
zgap=3;
%right solenoid configuration, 0 for none, 1 for oppsite current direction, -1 for same
%current direction
sright=-1;
%mu for iron core, set to 1 for no iron core case
mu=1;
%y position of the grid
y=0;
%------------constants------------
%constant mu0
mu0=4*pi*1e-7;
%minima distance
eps=1e-9;
%---------------------------------
%% Initialization parameters
C=mu*mu0*scurrent/(2*pi);
gridheight=ssize(1)*resolution;
gridwidth=(ssize(2)+slength)*resolution;

%------------progressbar------------
cpb = ConsoleProgressBar();
cpb.setLeftMargin(4);   % progress bar left margin
cpb.setTopMargin(1);    % rows margin
cpb.setLength(40);      % progress bar length: [.....]
cpb.setMinimum(0);      % minimum value of progress range [min max]
cpb.setMaximum(gridheight+5);    % maximum value of progress range [min max]
cpb.setPercentPosition('left');
cpb.setTextPosition('right');
cpb.start();
cpb.setValue(0);
%------------------------------------

%lefttop and rightbottom [x,y] for imagesc plot
xc=[0,ssize(2)];
yc=[ssize(1),-ssize(1)];
%x, z values, store in GPU
x=gpuArray(0:gridheight)/(1000*resolution);
z=gpuArray(0:gridwidth)/(1000*resolution);
%grid for B field calculation, store in GPU
bx=gpuArray(zeros(gridheight+1,gridwidth+1));
by=gpuArray(zeros(gridheight+1,gridwidth+1));
bz=gpuArray(zeros(gridheight+1,gridwidth+1));
bxy=gpuArray(zeros(gridheight+1,gridwidth+1));
%grid to store finial results
ssize=ssize*resolution;
Bx=gpuArray(zeros(ssize(1)+1,ssize(2)+1));
By=gpuArray(zeros(ssize(1)+1,ssize(2)+1));
Bz=gpuArray(zeros(ssize(1)+1,ssize(2)+1));
B=gpuArray(zeros(2*ssize(1)+1,ssize(2)+1));
%change to SI unit for calculation
sradius=sradius/1000;
zgap=zgap/1000;
%% Calculate B field in x,y,z direction

%move z by zgap
z=z+zgap;
rho=arrayfun(@sqrt,x.^2+y^2);   
for i=1:gridheight+1         
    r2=rho(i)^2+z.^2;
    alpha2=sradius^2+r2-2*sradius*rho(i);
    beta2=sradius^2+r2+2*sradius*rho(i);
    beta=arrayfun(@sqrt,beta2);
    k2=1-alpha2./beta2;
    [K,E]=ellipke(k2);
    bz(i,:)=C*((sradius^2-r2).*E + alpha2.*K)./(alpha2.*beta);
    if (rho(i)>=eps)         
        bxy(i,:)=C*z.*((sradius^2+r2).*E - alpha2.*K)./((rho(i)^2*alpha2.*beta));
    end    
    bx(i,:)=x(i)*bxy(i,:);
    by(i,:)=y*bxy(i,:);
    cpb.setValue(i);
end
%Calculate for N loops
for j=1:NumberOfTruns
    zp=round((j-1)*resolution*slength/NumberOfTruns)+1;
    Bx(:,:)=Bx(:,:)+bx(:,zp:zp+ssize(2));
    By(:,:)=By(:,:)+by(:,zp:zp+ssize(2));
    Bz(:,:)=Bz(:,:)+bz(:,zp:zp+ssize(2));
end
Bx(:,:)=Bx(:,:)/NumberOfTruns^2;
By(:,:)=By(:,:)/NumberOfTruns^2;
Bz(:,:)=Bz(:,:)/NumberOfTruns^2;
cpb.setValue(gridheight+5);
b=arrayfun(@sqrt,Bx.^2+By.^2+Bz.^2);
%Generate simulation data
B(1:gridheight+1,:)=flip(b,1);
B(gridheight+1:end,:)=b;
B=B+sright*flip(B,2);
B=gather(B);
cpb.stop();
%Clean and show results
clear alpha2 b beta beta2 bx by bz bxy cpb E eps i j K k2 mu0 r2 rho x z zp gridheight gridwidth sradius ssize C
imagesc(xc,yc,B)
set(gca,'YDir','normal')
[px,py]=gradient(B,1/resolution/1000,1/resolution/1000);
reduce=5*resolution;
reducedB=B(1:reduce:end,1:reduce:end);
reducedpx=px(1:reduce:end,1:reduce:end);
reducedpy=py(1:reduce:end,1:reduce:end);
figure
contour(reducedB)
hold on
quiver(reducedpx,reducedpy)
hold off