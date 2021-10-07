import random
import numpy as np
from maze import build_maze
from toolbox import random_policy
import matplotlib.pyplot as plt
import chrono


def create_maze(width, height, ratio):
    size = width * height
    n_walls = round(ratio * size)

    stop = False
    m = None
    # the loop below is used to check that the maze has a solution
    # if one of the values after value iteration is null, then another maze should be produced
    while not stop:
        walls = random.sample(range(size), int(n_walls))

        m = build_maze(width, height, walls)
        v, _ = value_iteration_v(m, render=False)
        if np.all(v):
            stop = True
    return m


def get_policy_from_q(q):
    # Outputs a policy given the action values
    policy=np.argmax(q,1)
    
    return policy


def get_policy_from_v(mdp, v):
    # Outputs a policy given the state values
    policy = np.zeros(mdp.nb_states)  # initial state values are set to 0
    n = 0
    for x in range(mdp.nb_states):  # for each state x
        # Compute the value of the state x for each action u of the MDP action space
        v_temp = []
        for u in mdp.action_space.actions:
            if x not in mdp.terminal_states:
                # Process sum of the values of the neighbouring states
                summ = 0
                for y in range(mdp.nb_states):
                    n+=1
                    summ = summ + mdp.P[x, u, y] * v[y]
                v_temp.append(mdp.r[x, u] + mdp.gamma * summ)
            else:  # if the state is final, then we only take the reward into account
                v_temp.append(mdp.r[x, u])
                
        policy[x] = np.argmax(v_temp)
    return policy, n


def improve_policy_from_v(mdp, v, policy):
    # Improves a policy given the state values
    n = 0
    for x in range(mdp.nb_states):  # for each state x
        # Compute the value of the state x for each action u of the MDP action space
        v_temp = np.zeros(mdp.action_space.size)
        for u in mdp.action_space.actions:
            if x not in mdp.terminal_states:
                # Process sum of the values of the neighbouring states
                summ = 0
                for y in range(mdp.nb_states):
                    n+=1
                    summ = summ + mdp.P[x, u, y] * v[y]
                v_temp[u] = mdp.r[x, u] + mdp.gamma * summ
            else:  # if the state is final, then we only take the reward into account
                v_temp[u] = mdp.r[x, u]

        for u in mdp.action_space.actions:
            
            if v_temp[u] > v_temp[int(policy[x])]:
                n+=1
                policy[x] = u
    return policy, n


def evaluate_one_step_v(mdp, v, policy):
    # Outputs the state value function after one step of policy evaluation
    # Corresponds to one application of the Bellman Operator
    v_new = np.zeros(mdp.nb_states)  # initial state values are set to 0
    n=0
    for x in range(mdp.nb_states):  # for each state x
        # Compute the value of the state x for each action u of the MDP action space
        v_temp = []
        if x not in mdp.terminal_states:
            # Process sum of the values of the neighbouring states
            summ = 0
            for y in range(mdp.nb_states):
                n+=1
                summ = summ + mdp.P[x, int(policy[x]), y] * v[y]
            v_temp.append(mdp.r[x, int(policy[x])] + mdp.gamma * summ)
        else:  # if the state is final, then we only take the reward into account
            v_temp.append(mdp.r[x, int(policy[x])])

        # Select the highest state value among those computed
        v_new[x] = np.max(v_temp)
    return v_new,n


def evaluate_v(mdp, policy):
    # Outputs the state value function of a policy
    v = np.zeros(mdp.nb_states)  # initial state values are set to 0
    stop = False
    n = 0
    while not stop:
        vold = v.copy()
        v , x= evaluate_one_step_v(mdp, vold, policy)
        n+=x
        # Test if convergence has been reached
        if (np.linalg.norm(v - vold)) < 0.01:
            stop = True
    return v,n


def evaluate_one_step_q(mdp, q, policy):
    # Outputs the state value function after one step of policy evaluation
    q_new=np.zeros((mdp.nb_states, mdp.action_space.size))
    n = 0
    for x in range(mdp.nb_states):
            for u in mdp.action_space.actions:
                if x in mdp.terminal_states:
                    q_new[x, :] = mdp.r[x, u]
                else:
                    # TODO: fill this
                    summ = 0
                    for y in range(mdp.nb_states):
                        n+=1
                        summ = summ + mdp.P[x, u, y] * q[y,policy[y]]
                    q_new[x, u] = mdp.r[x,u] + mdp.gamma*summ# TODO: fill this

    return q_new, n


def evaluate_q(mdp, policy):
    q = np.zeros((mdp.nb_states, mdp.action_space.size))  # initial state values are set to 0
    stop = False

    while not stop:
        qold = q.copy()
        q, n= evaluate_one_step_q(mdp, qold, policy)
        # Test if convergence has been reached
        if (np.linalg.norm(q - qold)) < 0.01:
            stop = True
    return q, n

# ------------------------- Value Iteration with the V function ----------------------------#
# Given a MDP, this algorithm computes the optimal state value function V
# It then derives the optimal policy based on this function
# This function is given


def value_iteration_v(mdp, render=True):
    # Value Iteration using the state value v
    v = np.zeros(mdp.nb_states)  # initial state values are set to 0
    v_list = []
    stop = False

    if render:
        mdp.new_render()
    i = 0
    n = 0
    c = chrono.Chrono()
    while not stop:
        v_old = v.copy()
        if render:
            mdp.render(v)

        for x in range(mdp.nb_states):  # for each state x
            # Compute the value of the state x for each action u of the MDP action space
            v_temp = []
            for u in mdp.action_space.actions:
                if x not in mdp.terminal_states:
                    # Process sum of the values of the neighbouring states
                    summ = 0
                    for y in range(mdp.nb_states):
                        summ = summ + mdp.P[x, u, y] * v_old[y]
                        n+=1
                    v_temp.append(mdp.r[x, u] + mdp.gamma * summ)
                else:  # if the state is final, then we only take the reward into account
                    v_temp.append(mdp.r[x, u])

                    # Select the highest state value among those computed
            v[x] = np.max(v_temp)

        # Test if convergence has been reached
        if (np.linalg.norm(v - v_old)) < 0.01:
            stop = True
        v_list.append(np.linalg.norm(v))
        i+=1

    c.stop()
    print("iteraction :",i)
    print("elementary operation:",n)
    if render:
        policy ,x= get_policy_from_v(mdp, v)
        mdp.render(v, policy)
    
    return v, v_list

# ------------------------- Value Iteration with the Q function ----------------------------#
# Given a MDP, this algorithm computes the optimal action value function Q
# It then derives the optimal policy based on this function


def value_iteration_q(mdp, render=True):
    q = np.zeros((mdp.nb_states, mdp.action_space.size))  # initial action values are set to 0
    q_list = []
    stop = False

    if render:
        mdp.new_render()
    i = 0
    n = 0
    c = chrono.Chrono()
    while not stop:
        qold = q.copy()

        if render:
            mdp.render(q)

        for x in range(mdp.nb_states):
            for u in mdp.action_space.actions:
                if x in mdp.terminal_states:
                    q[x, :] = mdp.r[x, u]
                else:
                    # TODO: fill this
                    summ = 0
                    for y in range(mdp.nb_states):
                        summ = summ + mdp.P[x, u, y] * max(qold[y])
                        n+=1
                    q[x, u] = mdp.r[x,u] + mdp.gamma*summ# TODO: fill this

        if (np.linalg.norm(q - qold)) <= 0.01:
            stop = True
        q_list.append(np.linalg.norm(q))
        i+=1
    c.stop()
    print("iteraction :",i)
    print("elementary operation:",n)
    if render:
        mdp.render(q)
        
    return q, q_list


# ------------------------- Policy Iteration with the Q function ----------------------------#
# Given a MDP, this algorithm simultaneously computes the optimal action value function Q and the optimal policy

def policy_iteration_q(mdp, render=True):  # policy iteration over the q function
    q = np.zeros((mdp.nb_states, mdp.action_space.size))  # initial action values are set to 0
    q_list = []
    policy = random_policy(mdp)

    stop = False

    if render:
        mdp.new_render()
    i = 0
    n = 0
    c = chrono.Chrono()
    while not stop:
        qold = q.copy()

        if render:
            mdp.render(q)

        # Step 1 : Policy evaluation
        # TODO: fill this$

        q, x=evaluate_q(mdp,policy)
        n += x
        policy=get_policy_from_q(q)
        n += 1
 
        # Step 2 : Policy improvement
        # TODO: fill this
        

        # Check convergence
        if (np.linalg.norm(q - qold)) <= 0.01:
            stop = True
        q_list.append(np.linalg.norm(q))
        i+=1
    c.stop()
    print("iteraction :",i)
    print("elementary operation:",n)
    if render:
        mdp.render(q, get_policy_from_q(q))
    return q, q_list


# ------------------------- Policy Iteration with the V function ----------------------------#
# Given a MDP, this algorithm simultaneously computes the optimal state value function V and the optimal policy

def policy_iteration_v(mdp, render=True):
    # policy iteration over the v function
    v = np.zeros(mdp.nb_states)  # initial state values are set to 0
    v_list = []
    policy = random_policy(mdp)

    stop = False

    if render:
        mdp.new_render()
    i = 0
    n = 0
    c = chrono.Chrono()
    while not stop:
        vold = v.copy()
        # Step 1 : Policy Evaluation
        # TODO: fill this
        v, x =evaluate_v(mdp, policy)
        n+=x
        policy, x=get_policy_from_v(mdp, v)
        n+=x
        if render:
            mdp.render(v)
            mdp.plotter.render_pi(policy)

        # Step 2 : Policy Improvement
        # TODO: fill this
        policy, x=improve_policy_from_v(mdp,v,policy)
        n+=x
        # Check convergence
        if (np.linalg.norm(v - vold)) < 0.01:
            stop = True
        v_list.append(np.linalg.norm(v))
        i+=1
    c.stop()
    print("iteraction :",i)
    print("elementary operation:",n)
    if render:
        mdp.render(v)
        mdp.plotter.render_pi(policy)
    return v, v_list

# -------- plot learning curves of Q-Learning and Sarsa using epsilon-greedy and softmax ----------#


def plot_convergence_vi_pi(m, render):
    v, v_list1 = value_iteration_v(m, render)
    q, q_list1 = value_iteration_q(m, render)
    v, v_list2 = policy_iteration_v(m, render)
    q, q_list2 = policy_iteration_q(m, render)

    plt.plot(range(len(v_list1)), v_list1, label='value_iteration_v')
    plt.plot(range(len(q_list1)), q_list1, label='value_iteration_q')
    plt.plot(range(len(v_list2)), v_list2, label='policy_iteration_v')
    plt.plot(range(len(q_list2)), q_list2, label='policy_iteration_q')

    plt.xlabel('Number of episodes')
    plt.ylabel('Norm of V or Q value')
    plt.legend(loc='upper right')
    plt.savefig("comparison_DP.png")
    plt.show()


def run_dyna_prog():
    walls = [7,8,9,10, 21, 27, 33, 32,31,30,45,46,47]
    height = 6
    width = 9
    render=False
    m = build_maze(width, height, walls)  # maze-like MDP definition
    #m = create_maze(10, 10, 0.35)
    print()
    plot_convergence_vi_pi(m, False)
    print("value iteration V")
    value_iteration_v(m, render)
    print("value iteration Q")
    value_iteration_q(m, render)
    print("policy iteration Q")
    policy_iteration_q(m, render)
    print("policy iteration V")
    policy_iteration_v(m, render)
    input("press enter")


if __name__ == '__main__':
    run_dyna_prog()
