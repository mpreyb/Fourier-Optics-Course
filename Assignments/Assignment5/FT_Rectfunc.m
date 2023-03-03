%  ------Fourier transform of rectangular function------
%     File name: FT_Rectfunc.m
%     Author: Maria Paula Rey, EAFIT University
%     Email: mpreyb@eafit.edu.co
%     Date last modified: 15/03/2022
%     Matlab Version: R2020b

clear
close all
T = 1;
t = -2.5 : 0.001 : 2.5;
x = rectpuls(t,T);
% subplot(2,1,1)
% plot(t,x,'r','Linewidth',3);
% axis([-2.5 2.5 0 1.2])
% title({'Rectangular Pulse'})
% xlabel({'Time(s)'});
% ylabel('Ampltude');
% grid

syms t omega
F(omega) = int(1*exp(1j*omega*t), t, -T/4, T/4);        % Rectangular function
figure
fplot(F, [-1 1]*16*pi)
grid



