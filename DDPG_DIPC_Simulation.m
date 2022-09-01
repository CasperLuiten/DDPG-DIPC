% Main script for simulating the control of the Double Inverted Pendulum on Cart (DIPC) using a
% DDPG agent. This script and provided functions are part of the report
% developed for AE4350 Bio-inspired Intelligence and Learning for Aerospace
% Applications, August 2022.
% Author:           Casper Luiten - C.J.Luiten@student.tudelft.nl
% Teaching staff:   Dr. G.C.H.E. de Croon - G.C.H.E.deCroon@tudelft.nl
%                   Dr.ir. E. van Kampen - E.vanKampen@tudelft.nl

close all
clear all

%% Get best agent from training results and create policy function
load('Agent1432')
saved_agent.generatePolicyFunction()

%% Parameters and Initialization
% Initial state
x0 = [-0.2; 0.01; -0.01; 0; 0; 0];

% Time array
t0 = 0;
tend = 20;
dt = 0.01;
t = linspace(t0, tend, 1 + round((tend-t0)/dt));

% State array
X = zeros(size(x0,1),length(t));
X(:,1) = x0;

% Inpute array
U = zeros(1,length(t));

% Create environment instance and validate environment
env = DIPC();
validateEnvironment(env)

%% Simulation
close(figure(1))

% Reference for tracking, set to zero for stabilisation
xref = zeros(1,length(t));
xref(8/dt:end) = 0.295;

% Set initial condition and plot environmnent
env.State = x0;
plot(env)

% Create video
myVideo = VideoWriter('myVideoFile.avi');
myVideo.FrameRate = 25;  
open(myVideo)

% Simulate episode and save correct number of frames
for k = 1:length(t)
    X(:,k) = env.State;
    U(k) = evaluatePolicy(X(:,k)-[xref(k); 0; 0; 0; 0; 0]);
    env.step(U(k));
    steps_sec = round(1/dt);
    if mod(t(k), 1/myVideo.FrameRate) == 0
        frame = getframe(figure(1));
        writeVideo(myVideo, frame);
    end
end
close(myVideo)
close(figure(1))

%% Plotting results
figure()
names = {'x [m]', '\theta_1 [deg]', '\theta_2 [deg]'};
for k = 1:size(X,1)-3
    subplot(4,1,k)
    if k > 1
        plot(t,X(k,:)/pi*180)
    else
        plot(t,xref,'Color', [0.35, 0.75, 0.2])
        hold on
        plot(t,X(k,:),'b')
        title(sprintf('x(0) = [%.1f, %.1f, %.1f, 0, 0, 0]',x0(1), x0(2)/pi*180, x0(3)/pi*180))
        legend('x_{ref}', 'x','Location', 'southeast','FontSize', 11.5)
    end
    xlim([t0,tend])
    grid on
    ylabel(names{k})
end
subplot(4,1,4)
plot(t,U,'r')
grid on
ylabel('u [N]')
xlabel('Time [sec]')
sgtitle('DDPG controlled DIPC')
