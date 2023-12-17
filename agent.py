import numpy as np
from scipy import stats # for gaussian noise
from environment import Environment

class DynaAgent(Environment):

    def __init__(self, alpha, gamma, epsilon):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha   -- learning rate \in (0, 1]
            gamma   -- discount factor \in (0, 1)
            epsilon -- controls the influence of the exploration bonus
        '''

        self.alpha   = alpha
        self.gamma   = gamma 
        self.epsilon = epsilon

        return None

    def init_env(self, env_config):

        '''
        Initialise the environment
        Input arguments:
            **env_config -- dictionary with environment parameters
        '''

        Environment.__init__(self, env_config)

        return None

    def _init_q_values(self):

        '''
        Initialise the Q-value table
        '''

        self.Q = np.zeros((self.num_states, self.num_actions))

        return None

    def _init_experience_buffer(self):

        '''
        Initialise the experience buffer
        '''

        self.experience_buffer = np.zeros((self.num_states*self.num_actions, 4), dtype=int)
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.experience_buffer[s*self.num_actions+a] = [s, a, 0, s]

        return None

    def _init_history(self):

        '''
        Initialise the history
        '''

        self.history = np.empty((0, 4), dtype=int)

        return None
    
    def _init_action_count(self):

        '''
        Initialise the action count
        '''

        self.action_count = np.zeros((self.num_states, self.num_actions), dtype=int)

        return None

    def _update_experience_buffer(self, s, a, r, s1):

        '''
        Update the experience buffer (world model)
        Input arguments:
            s  -- initial state
            a  -- chosen action
            r  -- received reward
            s1 -- next state
        '''
        """ Include the new s,a,r,s1 list in the experince buffer. """
        self.experience_buffer[s * self.num_actions + a, : ] = np.asarray([s, a, r, s1])
               
        return None

    def _update_qvals(self, s, a, r, s1, bonus=False):

        '''
        Update the Q-value table
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
            bonus -- True / False whether to use exploration bonus or not
        '''
        
        # complete the code
        """ Use the formula form the paper to update the q-values. Bonus is a boolean
            value to determine whether to use the exploration bonus or not. """
            
        """ Find the optimal action and its q-value for the next state. """
        optimal_action  = np.argmax(self.Q[s1,:])
        optimal_Q = self.Q[s1,optimal_action]

        if bonus:
            self.Q[s,a] += self.alpha * (r + (self.epsilon * np.sqrt(self.action_count[s,a])) + self.gamma* optimal_Q - self.Q[s,a])
        else:
            self.Q[s,a] += self.alpha * (r + self.gamma* optimal_Q - self.Q[s,a])
            
        return None

    def _update_action_count(self, s, a):

        '''
        Update the action count
        Input arguments:
            Input arguments:
            s  -- initial state
            a  -- chosen action
        '''

        # complete the code
        """ If I havn't done action incement it. If I have done specific action 
            in state then set it to 0. This shows me how long ago I have done
            specific action in specific state. """
        self.action_count = self.action_count + 1
        self.action_count[s,a] = 0
        
        
        return None

    def _update_history(self, s, a, r, s1):

        '''
        Update the history
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
        '''

        self.history = np.vstack((self.history, np.array([s, a, r, s1])))

        return None

    def _policy(self, s):

        '''
        Agent's policy 
        Input arguments:
            s -- state
        Output:
            a -- index of action to be chosen
        '''

        # complete the code
        # Epsilon-greedy policy
        
        """ Get the q-values for all actions in the current state."""
        Q_all_action = self.Q[s,:]
        """ Get the number of times each action has been taken in the current state. """
        number_states = self.action_count[s,:]
        """ Calculate the q-values with the exploration bonus. """
        Q_explore = Q_all_action + self.epsilon * np.sqrt(number_states)

        """ If not epsilon greedy is chosen. """
        if self.epsilon != -1:
            
            p_a_given_s = []
            
            """ Use softmax function to calulate the probability of each action."""                       
            for action in Q_all_action:
                p_a_given_s += [np.exp(self.alpha  * Q_explore[s,action]) / np.sum(self.alpha  * np.exp(Q_explore[s,a]) for a in Q_all_action)]
            
            """ Choose randomly an action with the probability of the softmax function. """
            a = np.random.choice(Q_explore, p = p_a_given_s)
        else: 
            """ Epsilon greedy algorithm """
            if np.random.uniform(0,1) < self.epsilon:
                """ Choose a random action. """
                a = np.random.randint(0,4)
            else:
                """ Choose the action with the highest q-value. """
                a = np.argmax(np.asarray([Q_explore]))
                         

        return a

    def _plan(self, num_planning_updates):

        '''
        Planning computations
        Input arguments:
            num_planning_updates -- number of planning updates to execute
        '''

        # complete the code
        """ For revisiting randomly old states. """
        for _ in range(num_planning_updates):

            """ Choose a random sars tuple from the experience buffer."""
            rand_sars = np.random.randint(self.experience_buffer.shape[0])
            s,a,r,s1 = self.experience_buffer[rand_sars]
            
            """ Update q-values with the random chosem sars tuple."""
            self._update_qvals(s, a, r, s1, bonus=True)
        return None

    def get_performace(self):

        '''
        Returns cumulative reward collected prior to each move
        '''

        return np.cumsum(self.history[:, 2])

    def simulate(self, num_trials, reset_agent=True, num_planning_updates=None):

        '''
        Main simulation function
        Input arguments:
            num_trials           -- number of trials (i.e., moves) to simulate
            reset_agent          -- whether to reset all knowledge and begin at the start state
            num_planning_updates -- number of planning updates to execute after every move
        '''

        if reset_agent:
            self._init_q_values()
            self._init_experience_buffer()
            self._init_action_count()
            self._init_history()

            self.s = self.start_state

        for _ in range(num_trials):

            # choose action
            a  = self._policy(self.s)
            # get into new state
            s1 = self._get_new_state(self.s, a)            
            # receive reward
            r  = self.R[self.s, a]
            # learning
            self._update_qvals(self.s, a, r, s1, bonus=False)
            # update world model 
            self._update_experience_buffer(self.s, a, r, s1)
            # reset action count
            self._update_action_count(self.s, a)
            # update history
            self._update_history(self.s, a, r, s1)
            # plan
            if num_planning_updates is not None:
                self._plan(num_planning_updates)

            if s1 == self.goal_state:
                self.s = self.start_state
            else:
                self.s = s1

        return None
    
class TwoStepAgent:

    def __init__(self, alpha1, alpha2, beta1, beta2, lam, w, p):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha1 -- learning rate for the first stage \in (0, 1]
            alpha2 -- learning rate for the second stage \in (0, 1]
            beta1  -- inverse temperature for the first stage
            beta2  -- inverse temperature for the second stage
            lam    -- eligibility trace parameter
            w      -- mixing weight for MF vs MB \in [0, 1] 
            p      -- perseveration strength
        '''

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1  = beta1
        self.beta2  = beta2
        self.lam    = lam
        self.w      = w
        self.p      = p

        return None
        
    def _init_history(self):

        '''
        Initialise history to later compute stay probabilities
        '''

        self.history = np.empty((0, 3), dtype=int)

        return None
    
    def _update_history(self, a, s1, r1):

        '''
        Update history
        Input arguments:
            a  -- first stage action
            s1 -- second stage state
            r1 -- second stage reward
        '''

        self.history = np.vstack((self.history, [a, s1, r1]))

        return None
    
    def get_stay_probabilities(self):

        '''
        Calculate stay probabilities
        '''

        common_r      = 0
        num_common_r  = 0
        common_nr     = 0
        num_common_nr = 0
        rare_r        = 0
        num_rare_r    = 0
        rare_nr       = 0
        num_rare_nr   = 0

        num_trials = self.history.shape[0]
        for idx_trial in range(num_trials-1):
            a, s1, r1 = self.history[idx_trial, :]
            a_next    = self.history[idx_trial+1, 0]

            # common
            if (a == 0 and s1 == 1) or (a == 1 and s1 == 2):
                # rewarded
                if r1 == 1:
                    if a == a_next:
                        common_r += 1
                    num_common_r += 1
                else:
                    if a == a_next:
                        common_nr += 1
                    num_common_nr += 1
            else:
                if r1 == 1:
                    if a == a_next:
                        rare_r += 1
                    num_rare_r += 1
                else:
                    if a == a_next:
                        rare_nr += 1
                    num_rare_nr += 1

        return np.array([common_r/num_common_r, rare_r/num_rare_r, common_nr/num_common_nr, rare_nr/num_rare_nr])

    def simulate(self, num_trials):

        '''
        Main simulation function
        Input arguments:
            num_trials -- number of trials to simulate
        '''
            
        # complete the code

        return None