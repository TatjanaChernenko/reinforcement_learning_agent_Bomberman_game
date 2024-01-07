
# Import stuff
import numpy as np
import matplotlib.pyplot as plt

def get_mean(flow, av):
    """ Return mean of flow over av steps.
    """
    return np.asarray([np.mean(flow[i:i+av],0) for i in range(flow.shape[0]-av+1)])
    
def get_epsilon_eff(choice, reward, av):
    """ Return epsilon_eff over av steps.
    """
    return 1-np.asarray([np.mean((np.sum(choice,1)/reward[:,19])[i:i+av],0) for i in range(choice.shape[0]-av+1)])

def get_action_share(reward, av=1):
    """ Return action share over av steps.
    """
    return np.asarray([np.mean((reward/reward[:,19:20]*100)[i:i+av],0) for i in range(reward.shape[0]-av+1)])

def get_forced_share(reward, av=1):
    """ Return forced action share over av steps.
    """
    return np.asarray([np.mean((reward[:,20]/reward[:,19]*100)[i:i+av],0) for i in range(reward.shape[0]-av+1)])

def get_memory_size(reward, offset=0):
    """ Return total memory size for every step.
    """
    return np.asarray([np.sum(x) + offset for x in [reward[:i+1,19] for i in range(reward.shape[0])]])

def get_memory(FLOW_A, offset=0):
    """ Return total memory size for every step.
    """
    return np.asarray([np.sum(x) + offset for x in [FLOW_A[1][0][:i+1,19] for i in range(FLOW_A[1][0].shape[0])]])

def get_compare(FLOW, VALUES, name="Random Chance"):
    """ Get random chance reward, coins, crates, length, points etc.
    """
    RANDOM = [np.mean(FLOW[1][0][:,17][:VALUES[1]]),np.mean(FLOW[1][0][:,11][:VALUES[1]]),np.mean(FLOW[1][0][:,9][:VALUES[1]])
             ,np.mean(FLOW[1][0][:,19][:VALUES[1]]),np.mean(FLOW[1][0][:,18][:VALUES[1]]),np.mean(FLOW[1][0][:,12][:VALUES[1]])
             ,np.mean(FLOW[1][0][:,13][:VALUES[1]]),np.mean(FLOW[1][0][:,14][:VALUES[1]])-np.mean(FLOW[1][0][:,13][:VALUES[1]])
             ,np.mean(FLOW[1][0][:,16][:VALUES[1]]),np.mean(FLOW[1][0][:,15][:VALUES[1]]),name]
    return RANDOM

def get_LIMIT_share(LIMIT, VALUES, index):
    share = np.count_nonzero(index)/VALUES[0]
    return [int(LIMIT[0]*share),int(LIMIT[1]*share)]

def get_VALUES_share(VALUES, index):
    share = np.count_nonzero(index)/VALUES[0]
    return [int(VALUES[0]*share),int(VALUES[1]*share)]+ VALUES[2:]

def get_length(save_perf):
    return np.load(save_perf+"/parameters_flow.npy").shape[0]

def get_FLOW(save_perf, index):
    """ Get list with reward, choice, param, action, total walk flow.
    """
    reward_flow = np.load(save_perf+"/reward_flow.npy")[index]
    choice_flow = np.load(save_perf+"/choice_flow.npy")[index]
    param_flow = np.load(save_perf+"/parameters_flow.npy")[index]
    action_share = reward_flow/reward_flow[:,19:20]*100
    total_walk = np.sum(choice_flow[:,:4],1)
    total_chosen = np.sum(choice_flow,1)
    epsilon_eff = 1 - total_chosen/reward_flow[:,19]
    return [reward_flow,choice_flow,param_flow,action_share,total_walk,epsilon_eff, total_chosen]

def get_means(list, av):
    """ Get list with means for every entry in list.
    """
    ret = []
    for flow in list:
        ret += [get_mean(flow, av)]
    return ret

def get_FLOW_A(FLOW, AV):
    ret = [AV, FLOW]
    for av in AV:
        ret += [get_means(FLOW,av)]
    return ret

def get_FALVR(save_perf, index, AV, LIMIT, VALUES):
    """ Get FLOW_A, LIMIT, VALUES, RANDOM for index crate etc.
    """
    FLOW_A_index = get_FLOW_A(get_FLOW(save_perf, index), AV)
    LIMIT_index = get_LIMIT_share(LIMIT, VALUES, index)
    VALUES_index = get_VALUES_share(VALUES, index)
    RANDOM_index = get_compare(FLOW_A_index, VALUES)
    return FLOW_A_index, LIMIT_index, VALUES_index, RANDOM_index

def plot_values(axis, values, style, label, offset=0):
    """ Plot array of values.
    """
    return axis.plot(np.arange(values.shape[0])+offset//2, values, style, label=label)

def plot_constant(axis, value, VALUES, style, label):
    """ Plot constant value.
    """
    return axis.plot(np.linspace(0, VALUES[0], 1000),np.full((1000),value),style,label=label)

def add_same_scale(axis):
    """ Add left y scale on the right.
    """
    axis.twinx().set_ylim(axis.get_ylim())

def label_last_axis(axis, label="Number of training rounds"):
    """ Add label below last axis.
    """
    axis.set_xlabel(label,size=14)

def plot_reward(axis, FLOW_A, LIMIT, VALUES, RANDOM=[], SIMPLE=[], YLIM=[None,None], LOC="best"):
    """ Plot for the cumulative reward. Axis must be part of plt.subplots!
    """
    plot_values(axis, FLOW_A[1][0][:,17], "b:", "Reward")
    plot_values(axis, FLOW_A[2][0][:,17], "g-", "Average Reward over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[3][0][:,17], "m-", "Average Reward over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    if len(RANDOM) > 0:
        plot_constant(axis,RANDOM[0],VALUES,"y--",RANDOM[10])
    if SIMPLE != []:
        plot_constant(axis,SIMPLE[0],VALUES,"c--","Simple Agent")
    axis.set_title("Reward",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    addscale = axis.twinx()
    addscale.set_ylim(axis.get_ylim())
    axis.set_ylabel("Cumulative reward per round",size=12)
    axis.axvline(VALUES[1])

def plot_length(axis, FLOW_A, LIMIT, VALUES, RANDOM=[], SIMPLE=[], YLIM=[0,None], LOC="best"):
    """ Plot for the length of one round. Axis must be part of plt.subplots!
    """
    plot_values(axis, FLOW_A[1][0][:,19], "b:", "Length")
    plot_values(axis, FLOW_A[2][0][:,19], "g-", "Average Length over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[3][0][:,19], "m-", "Average Length over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    if len(RANDOM) > 0:
        plot_constant(axis, RANDOM[3], VALUES, "y--", RANDOM[10])
    if SIMPLE != []:
        plot_constant(axis, SIMPLE[3], VALUES, "c--", "Simple Agent")
    if np.max(FLOW_A[1][0][:,19])>0.7*VALUES[2]:
        plot_constant(axis, VALUES[2], VALUES, "k:", "MAX")
    axis.set_title("Length of one Round",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Number of steps per round",size=12)
    axis.axvline(VALUES[1])

def plot_parameters(axis, FLOW_A, LIMIT, VALUES, CONFIG, YLIM=[[0,None],[0,None]], LOC="best"):
    """ Plot the parameters. Axis must be part of plt.subplots!
    """
    lns = []
    lns += plot_constant(axis, CONFIG[1][3], VALUES, "y--", "$\epsilon_{min}$")
    if CONFIG[0][5]:
        lns += plot_constant(axis, CONFIG[1][15], VALUES, "--", "$\epsilon_{BOMB\,min}$")
    lns += plot_values(axis, FLOW_A[1][2][:,0], "b-", "$\epsilon$")
    if CONFIG[0][5]:
        lns += plot_values(axis, FLOW_A[1][2][:,3], "-", "$\epsilon_{BOMB}$")
    lns += plot_values(axis, FLOW_A[3][5], "c-", "$\epsilon_{eff}\quad$"+" over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    if CONFIG[0][0]:
        lns += plot_values(axis, FLOW_A[1][2][:,1], "m-", "$PER_b$")
    axis_twin = axis.twinx()
    lns += plot_values(axis_twin, FLOW_A[1][2][:,2], "g-", "Learning Rate")
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM[0])
    axis_twin.set_ylim(YLIM[1])
    axis.legend(lns, [l.get_label() for l in lns], loc=LOC)
    axis.set_title("Parameters",size=20)
    axis.set_ylabel("$\epsilon,\quad PER_b$",size=12)
    axis_twin.set_ylabel("Learning Rate",size=12)
    axis.axvline(VALUES[1])

def plot_coins(axis, FLOW_A, LIMIT, VALUES, RANDOM=[], SIMPLE=[], YLIM=[0,None], LOC="best"):
    """ Plot collected coins per round.
    """
    plot_values(axis, FLOW_A[1][0][:,11], "b:", "Coins")
    plot_values(axis, FLOW_A[2][0][:,11], "g-", "Average Coins over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[3][0][:,11], "m-", "Average Coins over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    if len(RANDOM) > 0:
        plot_constant(axis, RANDOM[1], VALUES, "y--", RANDOM[10])
    if SIMPLE != []:
        plot_constant(axis, SIMPLE[1], VALUES, "c--", "Simple Agent")
    if np.max(FLOW_A[1][0][:,11]) > 0.7*9:
        plot_constant(axis, 9, VALUES, "k--", "MAX")
        YLIM=[0,10]
    axis.set_title("Coins",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Coins collected per round",size=12)
    axis.axvline(VALUES[1])

def plot_crates(axis, FLOW_A, LIMIT, VALUES, RANDOM=[], SIMPLE=[], YLIM=[0,None], LOC="best"):
    """ Plot destroyed crates per round.
    """
    plot_values(axis, FLOW_A[1][0][:,9], ":", "Crates")
    plot_values(axis, FLOW_A[2][0][:,9], "-", "Average Crates over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[3][0][:,9], "-", "Average Crates over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    if len(RANDOM) > 0:
        plot_constant(axis, RANDOM[2], VALUES, "--", RANDOM[10])
    if SIMPLE != []:
        plot_constant(axis, SIMPLE[2], VALUES, "--", "Simple Agent")
    if np.max(FLOW_A[1][0][:,9]) > 0.7*VALUES[3]:
        plot_constant(axis, VALUES[3], VALUES, "k--", "MAX")
    axis.set_title("Crates",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Crates destroyed per round",size=12)
    axis.axvline(VALUES[1])

def plot_points(axis, FLOW_A, LIMIT, VALUES, RANDOM=[], SIMPLE=[], YLIM=[0,None], LOC="best"):
    """ Plot collected coins per round.
    """
    plot_values(axis, FLOW_A[1][0][:,18], "b:", "Points")
    plot_values(axis, FLOW_A[2][0][:,18], "g-", "Average Points over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[3][0][:,18], "m-", "Average Points over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    if len(RANDOM) > 0:
        plot_constant(axis, RANDOM[4], VALUES, "y--", RANDOM[10])
    if SIMPLE != []:
        plot_constant(axis, SIMPLE[4], VALUES, "c--", "Simple Agent")
    if np.max(FLOW_A[1][0][:,18]) > 0.7*24:
        plot_constant(axis, 24, VALUES, "k--", "MAX")
    axis.set_title("Points",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Points earned per round",size=12)
    axis.axvline(VALUES[1])

def plot_kills(axis, FLOW_A, LIMIT, VALUES, RANDOM=[], SIMPLE=[], YLIM=[0,None], LOC="best"):
    """ Plot collected coins per round.
    """
    plot_values(axis, FLOW_A[1][0][:,12], "b:", "Kills")
    plot_values(axis, FLOW_A[2][0][:,12], "g-", "Average Kills over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[3][0][:,12], "m-", "Average Kills over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    if len(RANDOM) > 0:
        plot_constant(axis, RANDOM[5], VALUES, "y--", RANDOM[10])
    if SIMPLE != []:
        plot_constant(axis, SIMPLE[5], VALUES, "c--", "Simple Agent")
    if np.max(FLOW_A[1][0][:,12]) > 0.7*3:
        plot_constant(axis, 3, VALUES, "k--", "MAX")
    axis.set_title("Kills",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Kills per round",size=12)
    axis.axvline(VALUES[1])

def plot_suicides(axis, FLOW_A, LIMIT, VALUES, RANDOM=[], SIMPLE=[], YLIM=[0,1], LOC="best"):
    """ Plot collected coins per round.
    """
    plot_values(axis, FLOW_A[2][0][:,13], "g-", "Average Suicides over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[3][0][:,13], "m-", "Average Suicides over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    if len(RANDOM) > 0:
        plot_constant(axis, RANDOM[6], VALUES, "y--", RANDOM[10])
    if SIMPLE != []:
        plot_constant(axis, SIMPLE[6], VALUES, "c--", "Simple Agent")
    axis.set_title("Suicides",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Suicides per round",size=12)
    axis.axvline(VALUES[1])

def plot_killed(axis, FLOW_A, LIMIT, VALUES, RANDOM=[], SIMPLE=[], YLIM=[0,1], LOC="best"):
    """ Plot collected coins per round.
    """
    plot_values(axis, FLOW_A[2][0][:,14]-FLOW_A[2][0][:,13], "g-", "Average Killed over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[3][0][:,14]-FLOW_A[3][0][:,13], "m-", "Average Killed over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    if len(RANDOM) > 0:
        plot_constant(axis, RANDOM[7], VALUES, "y--", RANDOM[10])
    if SIMPLE != []:
        plot_constant(axis, SIMPLE[7], VALUES, "c--", "Simple Agent")
    axis.set_title("Killed",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Killed per round",size=12)
    axis.axvline(VALUES[1])

def plot_survived(axis, FLOW_A, LIMIT, VALUES, RANDOM=[], SIMPLE=[], YLIM=[0,1], LOC="best"):
    """ Plot collected coins per round.
    """
    plot_values(axis, FLOW_A[2][0][:,16], "g-", "Average Survived over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[3][0][:,16], "m-", "Average Survived over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    if len(RANDOM) > 0:
        plot_constant(axis, RANDOM[8], VALUES, "y--", RANDOM[10])
    if SIMPLE != []:
        plot_constant(axis, SIMPLE[8], VALUES, "c--", "Simple Agent")
    axis.set_title("Survived",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Survived per round",size=12)
    axis.axvline(VALUES[1])

def plot_opp_elim(axis, FLOW_A, LIMIT, VALUES, RANDOM=[], SIMPLE=[], YLIM=[0,None], LOC="best"):
    """ Plot collected coins per round.
    """
    plot_values(axis, FLOW_A[1][0][:,15], "b:", "Opponents eliminated")
    plot_values(axis, FLOW_A[2][0][:,15], "g-", "Average Opponents eliminated over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[3][0][:,15], "m-", "Average Opponents eliminated over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    if len(RANDOM) > 0:
        plot_constant(axis, RANDOM[9], VALUES, "y--", RANDOM[10])
    if SIMPLE != []:
        plot_constant(axis, SIMPLE[9], VALUES, "c--", "Simple Agent")
    if np.max(FLOW_A[1][0][:,15]) > 0.7*3:
        plot_constant(axis, 3, VALUES, "k--", "MAX")
    axis.set_title("Opponents eliminated",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Opponents eliminated per round",size=12)
    axis.axvline(VALUES[1])

def plot_crates_coins(axis, FLOW_A, LIMIT, VALUES, RANDOM=[], SIMPLE=[], YLIM=[[0,None],[0,10]], LOC="best"):
    """ Plot collected coins and destroyed crates per round.
    """
    axis_twin = axis.twinx()
    lns = []
    lns += plot_values(axis_twin, FLOW_A[1][0][:,11], "b:", "Coins")
    lns += plot_values(axis_twin, FLOW_A[2][0][:,11], "g-", "Average Coins over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    lns += plot_values(axis_twin, FLOW_A[3][0][:,11], "m-", "Average Coins over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    if len(RANDOM) > 0:
        lns += plot_constant(axis_twin, RANDOM[1], VALUES, "y--", "Points Random")
    if SIMPLE != []:
        lns += plot_constant(axis_twin, SIMPLE[1], VALUES, "c--", "Points Simple Agent")
    
    lns += plot_values(axis, FLOW_A[1][0][:,9], ":", "Crates")
    lns += plot_values(axis, FLOW_A[2][0][:,9], "-", "Average Crates over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    lns += plot_values(axis, FLOW_A[3][0][:,9], "-", "Average Crates over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    if len(RANDOM) > 0:
        lns += plot_constant(axis, RANDOM[2], VALUES, "--", "Crates Random")
    if SIMPLE != []:
        lns += plot_constant(axis, SIMPLE[2], VALUES, "--", "Crates Simple Agent")
    if np.max(FLOW_A[1][0][:,9]) > 0.7*VALUES[3]:
        lns += plot_constant(axis, VALUES[3], VALUES, "k--", "MAX Crates")
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM[0])
    axis_twin.set_ylim(YLIM[1])
    axis.legend(lns, [l.get_label() for l in lns], loc=LOC)
    axis.set_title("Crates & Coins",size=20)
    axis.set_ylabel("Crates destroyed per round",size=12)
    axis_twin.set_ylabel("Coins collected per round",size=12)
    axis.axvline(VALUES[1])

def plot_crates_coins_large(axis, FLOW_A, LIMIT, VALUES, RANDOM=[], SIMPLE=[], YLIM=[[0,None],[0,10]], LOC="best"):
    """ Plot collected coins and destroyed crates per round.
    """
    axis_twin = axis.twinx()
    lns = []
    lns += plot_values(axis_twin, FLOW_A[2][0][:,11], "g-", "Average Coins over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    lns += plot_values(axis_twin, FLOW_A[3][0][:,11], "m-", "Average Coins over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    if len(RANDOM) > 0:
        lns += plot_constant(axis_twin, RANDOM[1], VALUES, "y--", "Points Random")
    if SIMPLE != []:
        lns += plot_constant(axis_twin, SIMPLE[1], VALUES, "c--", "Points Simple Agent")
    
    lns += plot_values(axis, FLOW_A[2][0][:,9], "-", "Average Crates over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    lns += plot_values(axis, FLOW_A[3][0][:,9], "-", "Average Crates over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    if len(RANDOM) > 0:
        lns += plot_constant(axis, RANDOM[2], VALUES, "--", "Crates Random")
    if SIMPLE != []:
        lns += plot_constant(axis, SIMPLE[2], VALUES, "--", "Crates Simple Agent")
    if np.max(FLOW_A[2][0][:,9]) > 0.7*VALUES[3]:
        lns += plot_constant(axis, VALUES[3], VALUES, "k--", "MAX Crates")
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM[0])
    axis_twin.set_ylim(YLIM[1])
    axis.legend(lns, [l.get_label() for l in lns], loc=LOC)
    axis.set_title("Crates & Coins",size=20)
    axis.set_ylabel("Crates destroyed per round",size=12)
    axis_twin.set_ylabel("Coins collected per round",size=12)
    axis.axvline(VALUES[1])

def plot_memory(axis, MEMORY, LIMIT, VALUES, CONFIG, YLIM=[0,None], LOC="best"):
    """ Plot steps in memory.
    """
    if CONFIG[0][0]:
        MAX = CONFIG[2][0]
    else:
        MAX = CONFIG[3]
    max_name = ["MAX"]+[str(i+2)+" MAX" for i in range(np.floor(np.max(MEMORY)/MAX).astype(int))]
    times = np.ceil(np.max(MEMORY)/MAX).astype(int)-1
    for i in range(times+1):
        plot_constant(axis, MAX*(times-i+1), VALUES, "k-", max_name[times-i])
    plot_constant(axis, MEMORY[VALUES[1]], VALUES, "-", "RANDOM")
    plot_values(axis, MEMORY, "-", "Steps in Memory")
    axis.set_title("Memory",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Steps in Memory",size=12)
    axis.axvline(VALUES[1])

def plot_epsilon_eff(axis, FLOW_A, LIMIT, VALUES, CONFIG, YLIM=[0,None], LOC="best"):
    """ Plot epsilon_eff, large mean.
    """
    plot_constant(axis, CONFIG[1][3], VALUES, "y--", "$\epsilon_{min}$")
    plot_constant(axis, CONFIG[1][15], VALUES, "--", "$\epsilon_{BOMB\,min}$")
    plot_values(axis, FLOW_A[1][2][:,0], "b-", "$\epsilon$")
    plot_values(axis, FLOW_A[1][2][:,3], "-", "$\epsilon_{BOMB}$")
    plot_values(axis, FLOW_A[3][5], "c-", "$\epsilon_{eff}\quad$"+" over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.legend(loc=LOC)
    axis.set_title("Parameters",size=20)
    axis.set_ylabel("$\epsilon$",size=12)
    axis.axvline(VALUES[1])

def plot_chosen_actions(axis, FLOW_A, LIMIT, VALUES, YLIM=[0,None], LOC="best"):
    """ Plot chosen actions, large mean.
    """
    plot_values(axis, FLOW_A[3][6], "k-", "Average TOTAL over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][4], "c-", "Average WALK over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][0][:,20], "b-", "Average FORCED RANDOM over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][1][:,4], "m-", "Average WAIT over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][1][:,5], "y-", "Average BOMB over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    axis.set_title("Chosen Actions",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Actions chosen",size=12)
    axis.axvline(VALUES[1])

def plot_chosen_actions_both(axis, FLOW_A, LIMIT, VALUES, YLIM=[0,None], LOC="best"):
    """ Plot chosen actions, small and large mean.
    """
    plot_values(axis, FLOW_A[2][6], "k:", "Average TOTAL over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[2][4], "c:", "Average WALK over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[2][0][:,20], "b:", "Average FORCED RANDOM over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[2][1][:,4], "m:", "Average WAIT over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[2][1][:,5], "y:", "Average BOMB over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[3][6], "k-", "Average TOTAL over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][4], "c-", "Average WALK over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][0][:,20], "b-", "Average FORCED RANDOM over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][1][:,4], "m-", "Average WAIT over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][1][:,5], "y-", "Average BOMB over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    axis.set_title("Chosen Actions",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Actions chosen",size=12)
    axis.axvline(VALUES[1])

def plot_chosen_actions_small_both(axis, FLOW_A, LIMIT, VALUES, YLIM=[0,None], LOC="best"):
    """ Plot chosen actions, no and large mean.
        Above is better!
    """
    plot_values(axis, FLOW_A[1][6], "k:", "TOTAL")
    plot_values(axis, FLOW_A[1][4], "c:", "WALK")
    plot_values(axis, FLOW_A[1][0][:,20], "b:", "FORCED RANDOM")
    plot_values(axis, FLOW_A[1][1][:,4], "m:", "WAIT")
    plot_values(axis, FLOW_A[1][1][:,5], "y:", "BOMB")
    plot_values(axis, FLOW_A[3][6], "k-", "Average TOTAL over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][4], "c-", "Average WALK over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][0][:,20], "b-", "Average FORCED RANDOM over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][1][:,4], "m-", "Average WAIT over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][1][:,5], "y-", "Average BOMB over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    axis.set_title("Chosen Actions",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Actions chosen",size=12)
    axis.axvline(VALUES[1])

def plot_actions_share(axis, FLOW_A, LIMIT, VALUES, YLIM=[0,None], LOC="best"):
    """ Plot share of actions, large mean.
    """
    plot_values(axis, FLOW_A[3][3][:,0], "b-", "Average share of LEFT over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,1], "g-", "Average share of RIGHT over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,2], "r-", "Average share of UP over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,3], "c-", "Average share of DOWN over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,4], "m-", "Average share of WAIT over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,7], "y-", "Average share of BOMB over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,20], "k-", "Average share of FORCED RANDOM over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    axis.set_title("Actions Share",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Share of actions",size=12)
    axis.axvline(VALUES[1])

def plot_actions_share_both(axis, FLOW_A, LIMIT, VALUES, YLIM=[0,None], LOC="best"):
    """ Plot share of actions, large mean.
    """
    plot_values(axis, FLOW_A[2][3][:,0], "b:", "Average share of LEFT over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[2][3][:,1], "g:", "Average share of RIGHT over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[2][3][:,2], "r:", "Average share of UP over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[2][3][:,3], "c:", "Average share of DOWN over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[2][3][:,4], "m:", "Average share of WAIT over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[2][3][:,7], "y:", "Average share of BOMB over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[2][3][:,20], "k:", "Average share of FORCED RANDOM over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, FLOW_A[3][3][:,0], "b-", "Average share of LEFT over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,1], "g-", "Average share of RIGHT over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,2], "r-", "Average share of UP over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,3], "c-", "Average share of DOWN over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,4], "m-", "Average share of WAIT over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,7], "y-", "Average share of BOMB over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,20], "k-", "Average share of FORCED RANDOM over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    axis.set_title("Actions Share",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Share of actions",size=12)
    axis.axvline(VALUES[1])

def plot_actions_share_small_both(axis, FLOW_A, LIMIT, VALUES, YLIM=[0,None], LOC="best"):
    """ Plot share of actions, large mean.
    """
    plot_values(axis, FLOW_A[1][3][:,0], "b:", "Share of LEFT")
    plot_values(axis, FLOW_A[1][3][:,1], "g:", "Share of RIGHT")
    plot_values(axis, FLOW_A[1][3][:,2], "r:", "Share of UP")
    plot_values(axis, FLOW_A[1][3][:,3], "c:", "Share of DOWN")
    plot_values(axis, FLOW_A[1][3][:,4], "m:", "Share of WAIT")
    plot_values(axis, FLOW_A[1][3][:,7], "y:", "Share of BOMB")
    plot_values(axis, FLOW_A[1][3][:,20], "k:", "Share of FORCED RANDOM")
    plot_values(axis, FLOW_A[3][3][:,0], "b-", "Average share of LEFT over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,1], "g-", "Average share of RIGHT over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,2], "r-", "Average share of UP over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,3], "c-", "Average share of DOWN over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,4], "m-", "Average share of WAIT over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,7], "y-", "Average share of BOMB over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    plot_values(axis, FLOW_A[3][3][:,20], "k-", "Average share of FORCED RANDOM over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    axis.set_title("Actions Share",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Share of actions",size=12)
    axis.axvline(VALUES[1])

def plot_PARAM_MEM(FLOW_A, LIMIT, VALUES, MEMORY, CONFIG, save_perf, save=False, save_name="Parameters", size=(16.5,11.7)):
    """ PLot parameters, memory.
    """
    fig, axes = plt.subplots(2,1)
    fig.set_size_inches(size)
    
    plot_parameters(axes[0], FLOW_A, LIMIT, VALUES, CONFIG)
    plot_memory(axes[1], MEMORY, LIMIT, VALUES, CONFIG)
    label_last_axis(axes[1])
    
    if save: plt.savefig(save_perf+"/"+save_name+".pdf", dpi=150, format="pdf")

def plot_ONLY_CRATE(FLOW_A, LIMIT, VALUES, MEMORY, RANDOM, CONFIG, save_perf, SIMPLE=[], save=False, save_name="PerformanceOnlyCrate", both=False, size=(46.8,33.1)):
    """ Plot only crates.
    """
    fig, axes = plt.subplots(4,2)
    fig.set_size_inches(size)
    
    plot_reward(axes[0,0],FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_crates(axes[1,0], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_coins(axes[2,0], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_length(axes[3,0], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    label_last_axis(axes[3,0],"Number of training rounds with crates")
    
    plot_parameters(axes[0,1], FLOW_A, LIMIT, VALUES, CONFIG)
    plot_memory(axes[1,1], MEMORY, LIMIT, VALUES, CONFIG)
    if both:
        plot_chosen_actions_both(axes[2,1], FLOW_A, LIMIT, VALUES)
        plot_actions_share_both(axes[3,1], FLOW_A, LIMIT, VALUES)
    else:
        plot_chosen_actions(axes[2,1], FLOW_A, LIMIT, VALUES)
        plot_actions_share(axes[3,1], FLOW_A, LIMIT, VALUES)
    label_last_axis(axes[3,1],"Number of training rounds with crates")
    
    if save: plt.savefig(save_perf+"/"+save_name+".pdf", dpi=150, format="pdf")

def plot_ONLY_CRATE_LONG(FLOW_A, LIMIT, VALUES, MEMORY, RANDOM, CONFIG, save_perf, SIMPLE=[], save=False, save_name="PerformanceOnlyCrateLong", both=False, size=(23.4,66.2)):
    """ Plot only crates.
    """
    fig, axes = plt.subplots(8)
    fig.set_size_inches(size)
    
    plot_reward(axes[0],FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_crates(axes[1], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_coins(axes[2], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_length(axes[3], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    
    plot_parameters(axes[4], FLOW_A, LIMIT, VALUES, CONFIG)
    plot_memory(axes[5], MEMORY, LIMIT, VALUES, CONFIG)
    if both:
        plot_chosen_actions_both(axes[6], FLOW_A, LIMIT, VALUES)
        plot_actions_share_both(axes[7], FLOW_A, LIMIT, VALUES, LOC="upper left")
    else:
        plot_chosen_actions(axes[6], FLOW_A, LIMIT, VALUES)
        plot_actions_share(axes[7], FLOW_A, LIMIT, VALUES, LOC="upper left")
    label_last_axis(axes[7],"Number of training rounds with crates")
    
    if save: plt.savefig(save_perf+"/"+save_name+".pdf", dpi=150, format="pdf")

def plot_CRATE(FLOW_A, LIMIT, VALUES, RANDOM, CONFIG, save_perf, SIMPLE=[], save=False, save_name="PerformanceCrate", both=False, size=(46.8,33.1)):
    """ Plot crates.
    """
    fig, axes = plt.subplots(4,2)
    fig.set_size_inches(size)
    
    plot_reward(axes[0,0],FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_crates(axes[1,0], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_coins(axes[2,0], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_length(axes[3,0], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    label_last_axis(axes[3,0],"Number of training rounds with crates")
    
    plot_epsilon_eff(axes[0,1], FLOW_A, LIMIT, VALUES, CONFIG)
    if both:
        plot_chosen_actions_both(axes[2,1], FLOW_A, LIMIT, VALUES)
        plot_actions_share_both(axes[3,1], FLOW_A, LIMIT, VALUES)
    else:
        plot_chosen_actions(axes[2,1], FLOW_A, LIMIT, VALUES)
        plot_actions_share(axes[3,1], FLOW_A, LIMIT, VALUES)
    label_last_axis(axes[3,1],"Number of training rounds with crates")
    
    if save: plt.savefig(save_perf+"/"+save_name+".pdf", dpi=150, format="pdf")

def plot_COIN(FLOW_A, LIMIT, VALUES, RANDOM, CONFIG, save_perf, SIMPLE=[], save=False, save_name="PerformanceCoin", both=False, size=(33.1,23.4)):
    """ Plot coins.
    """
    fig, axes = plt.subplots(3,2)
    fig.set_size_inches(size)
    
    plot_reward(axes[0,0],FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_coins(axes[1,0], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_length(axes[2,0], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    label_last_axis(axes[2,0],"Number of training rounds with crates")
    
    plot_epsilon_eff(axes[0,1], FLOW_A, LIMIT, VALUES, CONFIG)
    if both:
        plot_chosen_actions_both(axes[1,1], FLOW_A, LIMIT, VALUES)
        plot_actions_share_both(axes[2,1], FLOW_A, LIMIT, VALUES)
    else:
        plot_chosen_actions(axes[1,1], FLOW_A, LIMIT, VALUES)
        plot_actions_share(axes[2,1], FLOW_A, LIMIT, VALUES)
    label_last_axis(axes[2,1],"Number of training rounds with crates")
    
    if save: plt.savefig(save_perf+"/"+save_name+".pdf", dpi=150, format="pdf")

def plot_crates_per_bomb(axis, FLOW_A, LIMIT, VALUES, RANDOM=[], SIMPLE=[], YLIM=[0,None], LOC="best"):
    """ Plot destroyed crates per bomb.
    """
    crates_per_bomb = FLOW_A[1][0][:,9]/FLOW_A[1][0][:,7]
    av, avl = FLOW_A[0]
    crates_per_bomb_av = np.asarray([np.mean(FLOW_A[1][0][i:i+av,9]/FLOW_A[1][0][i:i+av,7],0) for i in range(FLOW_A[1][0].shape[0]-av+1)])
    crates_per_bomb_avl = np.asarray([np.mean(FLOW_A[1][0][i:i+av,9]/FLOW_A[1][0][i:i+av,7],0) for i in range(FLOW_A[1][0].shape[0]-av+1)])
    plot_values(axis, crates_per_bomb, "b:", "Crates per bomb")
    plot_values(axis, crates_per_bomb_av, "g-", "Average Crates per bomb over {} Rounds".format(FLOW_A[0][0]), FLOW_A[0][0])
    plot_values(axis, crates_per_bomb_avl, "m-", "Average Crates per bomb over {} Rounds".format(FLOW_A[0][1]), FLOW_A[0][1])
    axis.set_title("Crates per bomb",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Crates destroyed per bomb",size=12)
    axis.axvline(VALUES[1])

def plot_fits(axis, FLOW_A, LIMIT, VALUES, RANDOM=[], SIMPLE=[], YLIM=[0,None], LOC="best"):
    """ Plot destroyed crates per bomb.
    """
    plot_values(axis, FLOW_A[1][0][:,21], "-", "Number of fits", offset=0)
    axis.set_title("Fits",size=20)
    axis.legend(loc=LOC)
    axis.set_xlim(LIMIT)
    axis.set_ylim(YLIM)
    add_same_scale(axis)
    axis.set_ylabel("Total number of fits",size=12)
    axis.axvline(VALUES[1])

def plot_ALL_LONG(FLOW_A, LIMIT, VALUES, MEMORY, RANDOM, CONFIG, save_perf, SIMPLE=[], save=False, save_name="PerformanceAllLong", both=False, size=(23.4,66.2)):
    """ Plot only crates.
    """
    fig, axes = plt.subplots(14)
    fig.set_size_inches(size)
    
    plot_reward(axes[0],FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_points(axes[1], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_kills(axes[2], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_crates(axes[3], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_coins(axes[4], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_opp_elim(axes[5], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_suicides(axes[6], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_killed(axes[7], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_survived(axes[8], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    plot_length(axes[9], FLOW_A, LIMIT, VALUES, RANDOM, SIMPLE)
    
    plot_parameters(axes[10], FLOW_A, LIMIT, VALUES, CONFIG)
    plot_memory(axes[11], MEMORY, LIMIT, VALUES, CONFIG)
    if both:
        plot_chosen_actions_both(axes[12], FLOW_A, LIMIT, VALUES)
        plot_actions_share_both(axes[13], FLOW_A, LIMIT, VALUES, LOC="upper left")
    else:
        plot_chosen_actions(axes[12], FLOW_A, LIMIT, VALUES)
        plot_actions_share(axes[13], FLOW_A, LIMIT, VALUES, LOC="upper left")
    label_last_axis(axes[13],"Number of training rounds with crates")
    
    if save: plt.savefig(save_perf+"/"+save_name+".pdf", dpi=150, format="pdf")






