import numpy as np
import random
import matplotlib.pyplot as plt

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

def phi(state, action):
    dealer_intervals = [(1,4), (4,7), (7,10)]
    player_intervals = [(1,6), (4,9), (7,12), (10,15), (13,18), (16,21)]
    actions = ["hit", "stick"]

    features = np.zeros((3, 6, 2))
    a_idx = actions.index(action)
    
    for i, d_range in enumerate(dealer_intervals):
        for j, p_range in enumerate(player_intervals):
            if d_range[0] <= state.dealer_card <= d_range[1] and p_range[0] <= state.player_card_sum <= p_range[1]:
                features[i, j, a_idx] = 1
    return features.flatten()  # 36-dimensional binary vector

theta = np.zeros(36)

def Q(state, action):
    return np.dot(theta, phi(state, action))


def epsilon_greedy_policy(state, epsilon=0.05):
    actions = ["hit", "stick"]
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        q_values = [Q(state, a) for a in actions]
        return actions[np.argmax(q_values)]


def sarsa_lambda(lam, n_episodes=1000, alpha=0.01, epsilon=0.05, gamma=1.0):
    global theta
    theta = np.zeros(36)
    mse_list = []

    for episode in range(n_episodes):
        state = State()
        action = epsilon_greedy_policy(state, epsilon)
        e = np.zeros(36)  # eligibility trace

        while not state.isTerminal:
            next_state, reward = step(state, action)
            next_action = None if next_state.isTerminal else epsilon_greedy_policy(next_state, epsilon)

            delta = reward - Q(state, action)
            if not next_state.isTerminal:
                delta += gamma * Q(next_state, next_action)

            e = gamma * lam * e + phi(state, action)
            theta += alpha * delta * e

            state = next_state
            action = next_action

        # track mean-squared error (for plotting)
        mse = np.mean(theta ** 2)
        mse_list.append(mse)

    return mse_list



mse_lambda_0 = sarsa_lambda(lam=0)
mse_lambda_1 = sarsa_lambda(lam=1)

plt.plot(mse_lambda_0, label='λ=0')
plt.plot(mse_lambda_1, label='λ=1')
plt.xlabel("Episode")
plt.ylabel("Mean Squared Error")
plt.title("SARSA(λ) with Linear Function Approximation in Easy21")
plt.legend()
plt.show()
