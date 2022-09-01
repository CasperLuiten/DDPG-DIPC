classdef DIPC < rl.env.MATLABEnvironment
    %DIPC: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Time step
        dt = 0.01
        
        % Physical parameters
        m_0 = 1.5
        m_1 = 0.75
        m_2 = 0.5
        L_1 = 0.75
        L_2 = 0.75
        g = 9.81

        % Input constraints
        umax = 5
        umin = -5

        % Bounds for plotting
        xmax = 3
        xmin = -3

        % Reset range
        delta = 0.05;
        
        % Angle at which to fail the episode (radians)
        AngleThreshold = 10/180*pi
        
        % Distance at which to fail the episode
        DisplacementThreshold = 2
        
        % Reward each time step the cart-pole is balanced
        RewardForNotFalling = 1
        Wx = diag([1 10 10 2 20 20])
        Wu = 0.1
        Wdu = 1;
        
        % Penalty when the cart-pole fails to balance
        PenaltyForFalling = 0
    end
    
    properties
        % Initialize system state and action placeholder
        State = zeros(6,1)
        LastAction = 0;
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false  
        
        % Handle to figure
        Figure
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = DIPC()
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([6 1]);
            ObservationInfo.Name = 'DIPC States';
            ObservationInfo.Description = 'x, theta1, theta2, dx, dtheta1, dtheta2';
            
            % Initialize Action settings   
            ActionInfo = rlNumericSpec([1 1]);
            ActionInfo.Name = 'DIPC Action';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            % Initialize property values and pre-compute necessary values
            updateActionInfo(this);
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
            LoggedSignals = [];
            
            % Get action
            Force = getForce(this,Action);            
            
            % Euler integration
            Observation = this.State + this.dt*SysDyn(this, Force);
            
            % Update system states and input
            this.State = Observation;
            
            % Check terminal condition
            X = Observation(1);
            Theta1 = Observation(2);
            Theta2 = Observation(3);
            IsDone = abs(X) > this.DisplacementThreshold || abs(Theta1) > this.AngleThreshold ...
                || abs(Theta2) > this.AngleThreshold;
            this.IsDone = IsDone;
            
            % Get reward
            Reward = getReward(this, Action);
            
            % Update action placeholder
            this.LastAction = Action;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            Theta2_0 = 2 * this.delta * rand - this.delta;  
            InitialObservation = [0; 0; Theta2_0; 0; 0; 0];
            this.State = InitialObservation;
            this.LastAction = 0;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods               
        % Helper methods to create the environment
        % Continuous force based on action
        function force = getForce(this,action)
%             if ~((action>this.ActionInfo.LowerLimit)&&(action<this.ActionInfo.UpperLimit))
%                 error('Action must be in allowable range');
%             end
            force = action;          
        end
        % update the action info based on max force
        function updateActionInfo(this)
            this.ActionInfo.UpperLimit = this.umax;
            this.ActionInfo.LowerLimit = this.umin;
        end
        
        % Reward function
        function Reward = getReward(this, Action)
            if ~this.IsDone
%                 Reward = this.RewardForNotFalling;
                Reward = 2*exp(-0.5*this.State'*this.Wx*this.State - 0.5*(this.Wu*Action^2)...
                             -0.5*this.Wdu*(Action-this.LastAction)^2);
%                 xTh = 0.5*[this.DisplacementThreshold; this.AngleThreshold; this.AngleThreshold;...
%                            1; 10/180*pi; 10/180*pi];
%                 uTh = 0.5*this.ActionInfo.UpperLimit;
%                 Reward = 0.5*(-this.State'*this.Wx*this.State + xTh'*this.Wx*xTh) + ...
%                          0.5*(-this.Wu*this.LastAction^2 + this.Wu*uTh^2);
            else
                Reward = this.PenaltyForFalling;
            end          
        end
        
        % Visualization method
        function plot(this)
            % Initiate the visualization
            this.Figure = figure('Visible','on','HandleVisibility','off');
            ha = gca(this.Figure);
            ha.XLimMode = 'manual';
            ha.YLimMode = 'manual';
            ha.XLim = [this.xmin-this.L_1-this.L_2-0.5, this.xmax+this.L_1+this.L_2+0.5];
            ha.YLim = [-this.L_1-this.L_2-0.5, this.L_1+this.L_2+0.5];
            hold(ha,'on');
            grid(ha,'on');
            % Update the visualization
            envUpdatedCallback(this)
        end
        
        % Compute model coefficients
        function [d_1, d_2, d_3, d_4, d_5, d_6, f_1, f_2] = getCoefficients(this)
            d_1 = this.m_0 + this.m_1 + this.m_2;
            d_2 = (0.5 * this.m_1 + this.m_2) * this.L_1;
            d_3 = 0.5 * this.m_2 * this.L_2;
            d_4 = (1/3 * this.m_1 + this.m_2) * this.L_1^2;
            d_5 = 0.5 * this.m_2 * this.L_1 * this.L_2;
            d_6 = 1/3 * this.m_2 * this.L_2^2;
            f_1 = (0.5 * this.m_1 + this.m_2 )*this.L_1*this.g;
            f_2 = 0.5 * this.m_2 * this.L_2 * this.g;
        end
        
        % System Dynamics
        function StateDot = SysDyn(this, Force)
            [d_1, d_2, d_3, d_4, d_5, d_6, f_1, f_2] = getCoefficients(this);
            
            % Compute once to prevent unnecessary computation time 
            CosS2 = cos(this.State(2));
            CosS3 = cos(this.State(3));
            CosS2mS3 = cos(this.State(2)-this.State(3));
            SinS2 = sin(this.State(2));
            SinS3 = sin(this.State(3));
            SinS2mS3 = sin(this.State(2)-this.State(3));
            
            D = [d_1,            d_2*CosS2,        d_3*CosS3;
                 d_2*CosS2,  d_4,                  d_5*CosS2mS3;
                 d_3*CosS3,  d_5*CosS2mS3,   d_6];

            C = [0,       -d_2*SinS2*this.State(5),        -d_3*SinS3*this.State(6);
                 0,        0,                              d_5*SinS2mS3*this.State(6);
                 0,       -d_5*SinS2mS3*this.State(5),    0];

            G = [0;
                 -f_1*SinS2;
                 -f_2*SinS3];

            H = [1; 
                 0; 
                 0];

            StateDot = [zeros(3), eye(3); zeros(3), -D^-1*C]*this.State + ...
                   [zeros(3,1); -D^-1*G] + [zeros(3,1); D^-1*H]*Force;
        end
    end
    
    methods (Access = protected)
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
            if ~isempty(this.Figure) && isvalid(this.Figure)
                % Set visualization figure as the current figure
                ha = gca(this.Figure);

                % Extract the cart position and pole angles
                x = this.State(1);
                u = this.LastAction;
                theta1 = this.State(2);
                theta2 = this.State(3);

                % Parameter arrow
                L_head = 0.2;
                if abs(u) < L_head
                    if u == 0
                        su = 1;
                    else
                        su = sign(u);
                    end
                    L_head = su*L_head;
                    L_arrow = su*0.01 + L_head;
                else
                    L_head = sign(u)*L_head;
                    L_arrow = u;
                end
                
                % Get objects
                cartplot = findobj(ha,'Tag','cartplot');
                pole1plot = findobj(ha,'Tag','pole1plot');
                pole2plot = findobj(ha,'Tag','pole2plot');
                forceplot = findobj(ha,'Tag','forceplot');
                if isempty(cartplot) || ~isvalid(cartplot) ...
                        || isempty(pole1plot) || ~isvalid(pole1plot)...
                        || isempty(pole2plot) || ~isvalid(pole2plot)...
                        || isempty(forceplot) || ~isvalid(forceplot)
                    % Initialize the cart plot
                    cartpoly = polyshape([-0.25 -0.25 0.25 0.25],[-0.125 0.125 0.125 -0.125]);
                    cartpoly = translate(cartpoly,[x 0]);
                    cartplot = plot(ha,cartpoly,'FaceColor',[0 0.7 0.1]);
                    cartplot.Tag = 'cartplot';

                    % Initialize the pole plots
                    L1 = this.L_1;
                    pole1poly = polyshape([-0.1 -0.1 0.1 0.1],[0 L1 L1 0]);
                    pole1poly = translate(pole1poly,[x,0]);
                    pole1poly = rotate(pole1poly,rad2deg(theta1),[x,0]);
                    pole1plot = plot(ha,pole1poly,'FaceColor',[0 0.4470 0.7410]);
                    pole1plot.Tag = 'pole1plot';
                    
                    L2 = this.L_2;
                    pole2poly = polyshape([-0.1 -0.1 0.1 0.1],[0 L2 L2 0]);
                    pole2poly = translate(pole2poly,[x,L1]);
                    pole2poly = rotate(pole2poly,rad2deg(theta2),[x,L1]);
                    pole2plot = plot(ha,pole2poly,'FaceColor',[0 0.4470 0.7410]);
                    pole2plot.Tag = 'pole2plot';

                    % Initialize force arrow
                    forcepoly = polyshape([0, L_arrow - L_head, L_arrow - L_head, L_arrow, L_arrow - L_head, L_arrow - L_head, 0],...
                                           [0.05, 0.05, 0.1, 0, -0.1, -0.05, -0.05]);
                    forcepoly = translate(forcepoly,[x,0]);
                    forceplot = plot(ha,forcepoly,'FaceColor',[0.7 0.001 0.001]);
                    forceplot.Tag = 'forceplot';
                else
                    cartpoly = cartplot.Shape;
                    pole1poly = pole1plot.Shape;
                    pole2poly = pole2plot.Shape;
                end

                % Compute the new cart and pole position
                % Step 1: Current x and y locations of centroids
                [cartx, ~] = centroid(cartpoly);
                [pole1x, pole1y] = centroid(pole1poly);
                [pole2x, pole2y] = centroid(pole2poly);
                
                % Step 2: Old theta1 and intersection point
                theta1_old = 0.5*pi - atan2(pole1y, (pole1x - cartx));
                ipx_old = pole1x + 0.5*this.L_1*sin(theta1_old);
                ipy_old = pole1y + 0.5*this.L_1*cos(theta1_old);
                
                % Step 3: Old theta2
                theta2_old = 0.5*pi - atan2((pole2y - ipy_old), (pole2x - ipx_old));
                
                % Step 4: New center points
                cp1x = x + 0.5*this.L_1*sin(theta1);
                cp1y = 0.5*this.L_1*cos(theta1);
                cp2x = x + this.L_1*sin(theta1) + 0.5*this.L_2*sin(theta2);
                cp2y = this.L_1*cos(theta1) + 0.5*this.L_2*cos(theta2);
                
                % Step 5: Changes
                dcartx = x - cartx;
                dcp1x = cp1x - pole1x;
                dcp1y = cp1y - pole1y;
                dcp2x = cp2x - pole2x;
                dcp2y = cp2y - pole2y;
                dtheta1 = theta1 - theta1_old;
                dtheta2 = theta2 - theta2_old;
                
                % Step 6: Update objects
                cartpoly = translate(cartpoly,[dcartx, 0]);
                pole1poly = translate(pole1poly, [dcp1x, dcp1y]);
                pole1poly = rotate(pole1poly, -rad2deg(dtheta1), [cp1x, cp1y]);
                pole2poly = translate(pole2poly, [dcp2x, dcp2y]);
                pole2poly = rotate(pole2poly, -rad2deg(dtheta2), [cp2x, cp2y]);

                % Step 7: Update force
                forcepoly = polyshape([0, L_arrow - L_head, L_arrow - L_head, L_arrow, L_arrow - L_head, L_arrow - L_head, 0],...
                                           [0.05, 0.05, 0.1, 0, -0.1, -0.05, -0.05]);
                forcepoly = translate(forcepoly,[x,0]);
                
                % Update the cart and pole positions on the plot
                cartplot.Shape = cartpoly;
                pole1plot.Shape = pole1poly;
                pole2plot.Shape = pole2poly;
                forceplot.Shape = forcepoly;

                % Refresh rendering in the figure window
                drawnow();
            end
        end
    end
end
