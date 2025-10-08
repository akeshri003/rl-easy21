import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm

class State:
    def __init__(self, dealer_card=None, player_card_sum=None, isTerminal = False):
        self.isTerminal = isTerminal
        if dealer_card is None:
            self.dealer_card = random.randint(1, 10)
        else:
            self.dealer_card = dealer_card
        
        if player_card_sum is None:
            self.player_card_sum = random.randint(1, 10)
        else:
            self.player_card_sum = player_card_sum


def drawCard() -> int:
    num = random.randint(1, 10)  # uniform integer between 1 and 10
    color = 1 if random.random() < (2/3) else -1  # 2/3 chance of +1, 1/3 chance of -1
    return num * color


def isBust(total: int) -> bool:
    return total < 1 or total > 21


def step(s: State, a: str) -> tuple[State, int]:

    dealer_card = s.dealer_card
    sum_player = s.player_card_sum

    if a == "stick":
        sum_dealer = dealer_card
        while sum_dealer < 17:
            new_card = drawCard()
            sum_dealer += new_card

        if isBust(sum_dealer): return (State(dealer_card, sum_player, True), 1)
        elif isBust(sum_player): return (State(dealer_card, sum_player, True), -1)
        else:
            if (21 - sum_player < 21 - sum_dealer):
                return (State(dealer_card, sum_player, True), 1)
            elif 21 - sum_player == 21 - sum_dealer:
                return (State(dealer_card, sum_player, True), 0)
            else:
                return (State(dealer_card, sum_player, True), -1)

    elif a == "hit":
        new_player_card = drawCard()
        sum_player += new_player_card

        if isBust(sum_player):
            return (State(dealer_card, sum_player, True), -1)
        else:
            return (State(dealer_card, sum_player, False), 0)


# ------- Monte Carlo Simulation -------
def epsilon_greedy_action(state, N0, N_s, Q):
    dealer, player = state.dealer_card, state.player_card_sum
    epsilon = N0 / (N0 + N_s[(dealer, player)])
    if random.random() < epsilon:
        return random.choice(["hit", "stick"])
    else:
        # pick best current Q
        q_hit, q_stick = Q[(dealer, player)]['hit'], Q[(dealer, player)]['stick']
        return "hit" if q_hit > q_stick else "stick"
    
def monte_carlo_control(num_episodes: int = 5000000, N0 : int = 100):
    st = State()
    Q = defaultdict(lambda: {'hit': 0.0, 'stick':0.0})
    N_sa = defaultdict(lambda: {'hit': 0, 'stick':0})
    N_s = defaultdict(int)

    for _ in tqdm(range(num_episodes), desc="Simulating Monte Carlo Episodes"):
        traj = []                       # [( (dealer,player), action ), ...]
        curr_state = State()

        # ----- generate one episode -----
        while True:
            d, p = curr_state.dealer_card, curr_state.player_card_sum
            N_s[(d, p)] += 1
            action = epsilon_greedy_action(curr_state, N0, N_s, Q)
            next_state, reward = step(curr_state, action)
            traj.append(((d, p), action))
            if next_state.isTerminal:
                G = reward              # <-- final return for MC
                break
            curr_state = next_state

        # ----- every-visit MC update -----
        for key, action in traj:
            N_sa[key][action] += 1
            alpha = 1.0 / N_sa[key][action]
            Q[key][action] += alpha * (G - Q[key][action])
    print("Num Q states learned:", len(Q))
    vals = [max(v["hit"], v["stick"]) for v in Q.values()]
    print("V* min/max:", (min(vals, default=0), max(vals, default=0)))

    V_star = {key: max(Q[key]["hit"], Q[key]["stick"]) for key in Q.keys()}
    return Q, V_star


# ----- SARSA (lamda) Learning -----

def sarsa_lambda(alpha: float, lmda : float, N0: int = 100, gamma : float = 1.0):
    Q = defaultdict(lambda: {'hit': 0.0, 'stick':0.0})
    N_s = defaultdict(int)

    for _ in tqdm(range(1000)):
        E = defaultdict(float)
        curr_state = State()
        action = epsilon_greedy_action(curr_state, N0, N_s, Q)

        while True:
            next_state, r = step(curr_state, action)
            next_act = epsilon_greedy_action(next_state, N0, N_s, Q)
            s_key = (curr_state.dealer_card, curr_state.player_card_sum)
            s1_key = (next_state.dealer_card, next_state.player_card_sum)
            a_key  = action
            if next_state.isTerminal:
                delta = r - Q[s_key][a_key]
            else:
                delta = r + gamma * Q[s1_key][next_act] - Q[s_key][a_key]
                E[(s_key, a_key)] = E[(s_key, a_key)] + 1
            
            for state in E:
                st_key, act = state
                Q[st_key][act] = Q[st_key][act] + alpha * delta * E[st_key][act]
                E[state] = gamma * lmda * E[state]

            curr_state = next_state
            action = next_act
            
            if next_state.isTerminal:
                break
        
    return Q

def sarsa(q_star, alpha: float, gamma: float = 1.0):
    lmda = 0.0
    error_cost = dict()
    while lmda <= 1.0:
        q = sarsa_lambda(alpha, lmda)
        mse = 0.0
        for key in q:
            for act in q[key]:
                if q_star[key][act] is not None:
                    mse += (q[key][act] - q_star[key][act]) ** 2
        
        error_cost[lmda] = mse
        lmda += 0.1

    return error_cost

# ----- Util -----
def plot_value_surface(V_star):
    """
    V_star: dict keyed by (dealer, player) -> value (float)
    dealer in [1..10], player in [1..21]
    """
    dealers = np.arange(1, 11)
    players = np.arange(1, 22)

    Z = np.full((len(players), len(dealers)), np.nan, dtype=float)
    for d in dealers:
        for p in players:
            Z[p-1, d-1] = V_star.get((d, p), np.nan)

    D, P = np.meshgrid(dealers, players)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(D, P, Z, linewidth=0, antialiased=True)

    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_zlabel("V*(s)")
    ax.set_title("Easy21 Optimal Value Function V*(s)")

    # Optional nicer viewing angle
    ax.view_init(elev=25, azim=-135)
    plt.tight_layout()
    plt.show()

def plot_greedy_policy(Q):
    dealers = np.arange(1, 11)
    players = np.arange(1, 22)
    policy = np.zeros((len(players), len(dealers)))  # 1=stick, 0=hit

    for d in dealers:
        for p in players:
            q = Q.get((d, p))
            if q is None:
                policy[p-1, d-1] = np.nan
            else:
                policy[p-1, d-1] = 1.0 if q["stick"] >= q["hit"] else 0.0

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(policy, origin='lower', aspect='auto',
                   extent=[1,10,1,21], interpolation='nearest')
    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_title("Greedy Policy: stick=1, hit=0")
    plt.tight_layout()
    plt.show()


Q, V_star = monte_carlo_control(num_episodes=5000000, N0=100)
plot_value_surface(V_star)
plot_greedy_policy(Q)


