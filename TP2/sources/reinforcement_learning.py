import numpy as np
from toolbox import softmax, egreedy_loc, egreedy
from maze import build_maze
import matplotlib.pyplot as plt

from dynamic_programming import policy_iteration_q, get_policy_from_q

# -------------------------------------------------------------------------------------#
# Given state and action spaces and a policy, computes the state value of this policy


def temporal_difference(mdp, pol, nb_episodes=50, alpha=0.2, timeout=25, render=True):
    # alpha: learning rate
    # timeout: timeout of an episode (maximum number of timesteps)
    v = np.zeros(mdp.nb_states)  # initial state value v
    mdp.timeout = timeout

    if render:
        mdp.new_render()
    
    for _ in range(nb_episodes):  # for each episode
        
        # Draw an initial state randomly (if uniform is set to False, the state is drawn according to the P0 
        #                                 distribution)
        x = mdp.reset(uniform=True) 
        done = mdp.done()
        while not done:  # update episode at each timestep
            # Show agent
            if render:
                mdp.render(v, pol)
            
            # Step forward following the MDP: x=current state, 
            #                                 pol[i]=agent's action according to policy pol, 
            #                                 r=reward gained after taking action pol[i], 
            #                                 done=tells whether the episode ended, 
            #                                 and info gives some info about the process
            [y, r, done, _] = mdp.step(egreedy_loc(pol[x], mdp.action_space.size, epsilon=0.2))
            
            # Update the state value of x
            if x in mdp.terminal_states:
                v[x] = mdp.r[x,pol[x]] #TODO: fill this
            else:
                delta = mdp.r[x,pol[x]] + mdp.gamma*v[y] - v[x]#TODO: fill this
                v[x] = v[x] + alpha*delta#TODO: fill this
            
            # Update agent's position (state)
            x = y
    
    if render:
        # Show the final policy
        mdp.current_state = 0
        mdp.render(v, pol)
    return v

# --------------------------- Q-Learning -------------------------------#

# Given a temperature "tau", the QLearning function computes the state action-value function
# based on a softmax policy
# alpha is the learning rate


def q_learning_soft(mdp, tau, nb_episodes=20, timeout=50, alpha=0.5, render=True):
    # Initialize the state-action value function
    # alpha is the learning rate
    q = np.zeros((mdp.nb_states, mdp.action_space.size))
    q_min = np.zeros((mdp.nb_states, mdp.action_space.size))

    q_list = []

    # Run learning cycle
    mdp.timeout = timeout  # episode length

    if render:
        mdp.new_render()

    for i in range(nb_episodes):
        # Draw the first state of episode i using a uniform distribution over all the states
        x = mdp.reset(uniform=True)
        done = mdp.done()
        while not done:
            if render:
                # Show the agent in the maze
                mdp.render(q, q.argmax(axis=1))

            # Draw an action using a soft-max policy
            u = mdp.action_space.sample(prob_list=softmax(q, x, tau))

            # Perform a step of the MDP
            [y, r, done, _] = mdp.step(u)

            # Update the state-action value function with q-Learning
            if x in mdp.terminal_states:
                q[x, u] = mdp.r[x,u]#TODO: fill this
            else:
                delta = mdp.r[x,u] + mdp.gamma*np.max(q[y]) - q[x, u] #TODO: fill this
                q[x, u] = q[x, u] + alpha*delta#TODO: fill this

            # Update the agent position
            x = y

        q_list.append(np.linalg.norm(np.maximum(q, q_min)))


    if render:
        # Show the final policy
        mdp.current_state = 0
        mdp.render(q, get_policy_from_q(q))
    return q, q_list


def q_learning_eps(mdp, eps, nb_episodes=20, timeout=50, alpha=0.5, render=True):
    # Initialize the state-action value function
    # alpha is the learning rate
    q = np.zeros((mdp.nb_states, mdp.action_space.size))
    q_min = np.zeros((mdp.nb_states, mdp.action_space.size))
    q_list = []

    # Run learning cycle
    mdp.timeout = timeout  # episode length

    if render:
        mdp.new_render()

    for _ in range(nb_episodes):
        # Draw the first state of episode i using a uniform distribution over all the states
        x = mdp.reset(uniform=True)
        done = mdp.done()
        while not done:
            if render:
                # Show the agent in the maze
                mdp.render(q, q.argmax(axis=1))

            # Draw an action using a soft-max policy
            u = egreedy(q, x, eps)

            # Perform a step of the MDP
            [y, r, done, _] = mdp.step(u)

            # Update the state-action value function with q-Learning
            if x in mdp.terminal_states:
                q[x, u] = mdp.r[x,u]#TODO: fill this
            else:
                delta = mdp.r[x,u] + mdp.gamma*np.max(q[y]) - q[x, u] #TODO: fill this
                q[x, u] = q[x, u] + alpha*delta#TODO: fill this

            # Update the agent position
            x = y
        q_list.append(np.linalg.norm(np.maximum(q, q_min)))

    if render:
        # Show the final policy
        mdp.current_state = 0
        mdp.render(q, get_policy_from_q(q))
    return q, q_list

def sarsa_soft(mdp, tau, nb_episodes=20, timeout=50, alpha=0.5, render=True):
    # Initialize the state-action value function
    # alpha is the learning rate
    q = np.zeros((mdp.nb_states, mdp.action_space.size))
    q_min = np.zeros((mdp.nb_states, mdp.action_space.size))
    q_list = []

    # Run learning cycle
    mdp.timeout = timeout  # episode length

    if render:
        mdp.new_render()

    for _ in range(nb_episodes):
        # Draw the first state of episode i using a uniform distribution over all the states
        x = mdp.reset(uniform=True)
        done = mdp.done()
        while not done:
            if render:
                # Show the agent in the maze
                mdp.render(q, q.argmax(axis=1))

            # Draw an action using a soft-max policy
            u = mdp.action_space.sample(prob_list=softmax(q, x, tau))

            # Perform a step of the MDP
            [y, r, done, _] = mdp.step(u)
            
            
            
            # Update the state-action value function with q-Learning
            if x in mdp.terminal_states:
                q[x, u] = mdp.r[x,u]#TODO: fill this
            else:
                v = mdp.action_space.sample(prob_list=softmax(q, y, tau))
                if mdp.r[x,u] != r:
                    print(mdp.r[x,u], r)
                delta = mdp.r[x,u] + mdp.gamma*q[y,v] - q[x, u] #TODO: fill this
                q[x, u] = q[x, u] + alpha*delta#TODO: fill this

            # Update the agent position
            x = y
        q_list.append(np.linalg.norm(np.maximum(q, q_min)))

    if render:
        # Show the final policy
        mdp.current_state = 0
        mdp.render(q, get_policy_from_q(q))
    return q, q_list

def sarsa_eps(mdp, eps, nb_episodes=20, timeout=50, alpha=0.5, render=True):
    # Initialize the state-action value function
    # alpha is the learning rate
    q = np.zeros((mdp.nb_states, mdp.action_space.size))
    q_min = np.zeros((mdp.nb_states, mdp.action_space.size))
    q_list = []

    # Run learning cycle
    mdp.timeout = timeout  # episode length

    if render:
        mdp.new_render()

    for _ in range(nb_episodes):
        # Draw the first state of episode i using a uniform distribution over all the states
        x = mdp.reset(uniform=True)
        done = mdp.done()
        while not done:
            if render:
                # Show the agent in the maze
                mdp.render(q, q.argmax(axis=1))

            # Draw an action using a soft-max policy
            u = egreedy(q, x, eps)

            # Perform a step of the MDP
            [y, r, done, _] = mdp.step(u)
            
 
            # Update the state-action value function with q-Learning
            if x in mdp.terminal_states:
                q[x, u] = mdp.r[x,u]#TODO: fill this
            else:
                v = egreedy(q, y, eps)
                delta = mdp.r[x,u] + mdp.gamma*q[y,v] - q[x, u] #TODO: fill this
                q[x, u] = q[x, u] + alpha*delta#TODO: fill this

            # Update the agent position
            x = y
        q_list.append(np.linalg.norm(np.maximum(q, q_min)))

    if render:
        # Show the final policy
        mdp.current_state = 0
        mdp.render(q, get_policy_from_q(q))
    return q, q_list
# -------- plot learning curves of Q-Learning and Sarsa using epsilon-greedy and softmax ----------#

def plot_ql_sarsa(m, epsilon, tau, nb_episodes, timeout, alpha, render):
    
    q, q_list1 = q_learning_eps(m, epsilon, nb_episodes, timeout, alpha, render)
    
    q, q_list2 = q_learning_soft(m, 0.1, nb_episodes, timeout, alpha, render)
    q, q_list3 = q_learning_eps(m, epsilon*10, nb_episodes, timeout, alpha, render)
    q, q_list4 = q_learning_soft(m, 5, nb_episodes, timeout, alpha, render)
    q, q_list5 = q_learning_eps(m, epsilon*100, nb_episodes, timeout, alpha, render)
    q, q_list6 = q_learning_soft(m, 10, nb_episodes, timeout, alpha, render)
    
    q, q_list7 = sarsa_eps(m, epsilon, nb_episodes, timeout, alpha, render)
    q, q_list8 = sarsa_soft(m, 0.1, nb_episodes, timeout, alpha, render)
    q, q_list9 = sarsa_eps(m, epsilon*10, nb_episodes, timeout, alpha, render)
    q, q_list10 = sarsa_soft(m, 5, nb_episodes, timeout, alpha, render)
    q, q_list11= sarsa_eps(m, epsilon*100, nb_episodes, timeout, alpha, render)
    q, q_list12= sarsa_soft(m, 10, nb_episodes, timeout, alpha, render)
    
    plt.plot(range(len(q_list1)), q_list1, label='q-learning epsilon'+str(epsilon),linewidth=0.5)
    plt.plot(range(len(q_list2)), q_list2, label='q-learning tau 0.1')
    plt.plot(range(len(q_list3)), q_list3, label='q-learning epsilon'+str(epsilon*10),linewidth=0.5)
    plt.plot(range(len(q_list4)), q_list4, label='q-learning tau 5')
    plt.plot(range(len(q_list5)), q_list5, label='q-learning epsilon'+str(epsilon*100),linewidth=0.5)
    plt.plot(range(len(q_list6)), q_list6, label='q-learning tau 10',linewidth=0.5)
    plt.plot(range(len(q_list7)), q_list7, label='sarsa epsilon '+str(epsilon),linewidth=0.5)
    plt.plot(range(len(q_list8)), q_list8, label='sarsa tau 0.1',linewidth=0.5)
    plt.plot(range(len(q_list9)), q_list9, label='sarsa epsilon '+str(epsilon*10),linewidth=0.5)
    plt.plot(range(len(q_list10)), q_list10, label='sarsa tau 5',linewidth=0.5)
    plt.plot(range(len(q_list11)), q_list11, label='sarsa epsilon '+str(epsilon*100),linewidth=0.5)
    plt.plot(range(len(q_list12)), q_list12, label='sarsa tau 10',linewidth=0.5)

    plt.xlabel('Number of episodes')
    plt.ylabel('Norm of Q values')
    plt.legend(title='legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("comparison_RL.png")
    plt.show()

# --------------------------- run it -------------------------------#


def run_rl():
    walls = [5, 6, 13]
    height = 4
    width = 5
    m = build_maze(width, height, walls, hit=True)

    q, x= policy_iteration_q(m, render=False)
    pol = get_policy_from_q(q)
    #print("TD-learning")
    #temporal_difference(m, pol, render=True)
    #print("Q-learning")
    #q_learning_soft(m, tau=5,nb_episodes=100)
    
    #print("Q-learning_soft")
    #q_learning_eps(m, eps=0.01)
    #sarsa_soft(m, tau=0.1,nb_episodes=50)
    #sarsa_eps(m, eps=0.1,nb_episodes=50)
    plot_ql_sarsa(m, 0.001, 6, 10000, 50, 0.5, False)
    #plot_ql_sarsa(m, 0.01, 6, 10000, 50, 0.5, False)

    #input("press enter")



if __name__ == '__main__':
    run_rl()
