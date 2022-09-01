% Main script for training of DDPG agent for the control of the Double 
% Inverted Pendulum on Cart (DIPC). This script and provided functions are 
% part of the report developed for AE4350 Bio-inspired Intelligence and 
% Learning for Aerospace Applications, August 2022.
% Author:           Casper Luiten - C.J.Luiten@student.tudelft.nl
% Teaching staff:   Dr. G.C.H.E. de Croon - G.C.H.E.deCroon@tudelft.nl
%                   Dr.ir. E. van Kampen - E.vanKampen@tudelft.nl

close all
clear all

%% Environment 
% Create environment object
env = DIPC();

% Validate the environment
validateEnvironment(env)

% Extract observation and action info
observationInfo = getObservationInfo(env);
actionInfo = getActionInfo(env);

%% Actor and Critic 
% Critic structure
statePath = [
    featureInputLayer(observationInfo.Dimension(1),'Normalization','none','Name','observation')
    fullyConnectedLayer(400,'Name','CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(300,'Name','CriticStateFC2')];
actionPath = [
    featureInputLayer(1,'Normalization','none','Name','action')
    fullyConnectedLayer(300,'Name','CriticActionFC1')];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu1')
    fullyConnectedLayer(1,'Name','CriticOutput')];

% Connect layers
criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
 
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
criticNetwork = dlnetwork(criticNetwork);

% Create critic
criticOpts = rlOptimizerOptions('LearnRate', 0.01,'GradientThreshold', inf);
critic = rlQValueFunction(criticNetwork,observationInfo,actionInfo,'ObservationInputNames','observation','ActionInputNames','action');

% Actor network
actorNetwork = [
    featureInputLayer(observationInfo.Dimension(1),'Normalization','none','Name','observation')
    fullyConnectedLayer(400,'Name','ActorFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(300,'Name','ActorFC2')
    reluLayer('Name','ActorRelu2')
    fullyConnectedLayer(1,'Name','ActorFC4')
    tanhLayer('Name','ActorTanh')
    scalingLayer('Name','ActorScaling','Scale', actionInfo.UpperLimit)];
actorNetwork = dlnetwork(actorNetwork);

% Create actor
actorOpts = rlOptimizerOptions('LearnRate', 0.01,'GradientThreshold',inf);
actor = rlContinuousDeterministicActor(actorNetwork,observationInfo,actionInfo);

%% Agent
% Option to select previously trained agent to train once more
USE_PRE_TRAINED_MODEL = false;

% Agent options
agentOpts = rlDDPGAgentOptions(...
    'SampleTime',env.dt,...
    'CriticOptimizerOptions',criticOpts,...
    'ActorOptimizerOptions',actorOpts,...
    'ExperienceBufferLength',1e6,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',128);
agentOpts.NoiseOptions.StandardDeviation = 0.3;
agentOpts.NoiseOptions.StandardDeviationDecayRate = 0;
agentOpts.TargetSmoothFactor = 1e-5;
agentOpts.ResetExperienceBufferBeforeTraining = not(USE_PRE_TRAINED_MODEL);

% Create agent
if USE_PRE_TRAINED_MODEL   
	disp('Continue training pre-trained model')
    preAgentName = uigetfile('*.mat');
    agent = load(preAgentName).saved_agent;
else
    agent = rlDDPGAgent(actor, critic, agentOpts);
end

%% Training
% Set options
opt = rlTrainingOptions;
opt.MaxEpisodes = 2000;
opt.MaxStepsPerEpisode = 2000;
opt.StopTrainingCriteria = "AverageReward";
opt.StopTrainingValue = 1500;
opt.SaveAgentCriteria = "EpisodeReward";
opt.SaveAgentValue = 1500;
opt.Verbose = false;
opt.ScoreAveragingWindowLength = 5;
opt.Plots = "training-progress";

% Execute training
close(figure(1))
plot(env)
trainResults = train(agent,env,opt);

% Automatically save final agent
if USE_PRE_TRAINED_MODEL
    saveName = append('Agent', num2str(trainResults.EpisodeIndex(end) + ...
        str2num( regexprep( preAgentName, {'\D*([\d\.]+\d)[^\d]*', '[^\d\.]*'}, {'$1 ', ' '} ) )));
else
    saveName = append('Agent',num2str(trainResults.EpisodeIndex(end)));
end
saved_agent = agent;
savedAgentResults = trainResults;
save(append('savedAgents/',saveName), 'saved_agent','savedAgentResults')

