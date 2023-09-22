import pandas as pd
import gymnasium as gym
import numpy as np
import math
from gymnasium import spaces
from gymnasium.envs.registration import register


class StockEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, price_data, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode
        self._price_data = price_data  # pd.DataFrame
        # Initial value required for the first observation
        self._cash = 0.0
        self._share_price = 0.0
        self._share_value = 0.0
        self.trade_history = {
            "day": [],
            "cash": [],
            "shares_owned": [],
            "owned_shares_value": [],
            "total_value": [],
            "action_selected": [],
        }

        # State: cash_available, shares_owned_no, share_price, share_value, total_value
        high = np.array(
            [
                np.finfo(np.float32).max,  # cash
                np.finfo(np.float32).max,  # share_price
                np.iinfo(np.int32).max,  # share_value
            ],
            dtype=np.float32,
        )

        low = np.zeros(high.size, dtype=np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Action space:
        # buy_little - buy stock worth 1% of initial cash available,
        # buy_moderate - buy stock worth 10% of initial cash available,
        # buy_lots - buy stock worth 20% of initial cash available,
        # sell_little - sell stock worth 1% of initial cash available,
        # sell_moderate - sell stock worth 10% of initial cash available,
        # sell_lots - sell stock worth 20% of initial cash available,
        # hold - do nothing
        self.action_space = spaces.Discrete(7)

    def _add_trade_to_history(
        self, day, cash, shares_owned, owned_shares_value, total_value, action_selected
    ):
        self.trade_history["day"].append(day)
        self.trade_history["cash"].append(cash)
        self.trade_history["shares_owned"].append(shares_owned)
        self.trade_history["owned_shares_value"].append(owned_shares_value)
        self.trade_history["total_value"].append(total_value)
        self.trade_history["action_selected"].append(action_selected)

    def _get_obs(self):
        return (
            np.array(
                [
                    self._cash,
                    self._share_price,
                    self._share_value,
                ],
                dtype=np.float32,
            ),
            {},
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
        cash_start=10000.0,
    ):
        super().reset(seed=seed, options=options)
        self._step_no = 0
        self._cash = cash_start
        self._shares_owned_no = 0.0
        self._share_price = 0.0
        self._share_value = 0.0
        self._total_value = self._cash + self._share_value
        self._current_price_data = self._price_data.iloc[0]
        self._share_price = self._current_price_data["Adj Close"]

        return self._get_obs()

    def action_to_percent_map(self):
        return {
            0: 0.01,
            1: 0.1,
            2: 0.2,
            3: 0.01,
            4: 0.1,
            5: 0.2,
            6: 0,
        }

    def action_to_name_map(self):
        return {
            0: "Buy little",
            1: "Buy moderate",
            2: "Buy lots",
            3: "Sell little",
            4: "Sell moderate",
            5: "Sell lots",
            6: "Hold",
        }

    def calc_max_shares_to_buy(self, action):
        return math.floor(
            self.action_to_percent_map()[action] * self._cash / self._share_price
        )

    def _get_action_meaning(self, action):
        return (
            self.action_to_name_map()[action]
            + f" ({self.action_to_percent_map()[action] * 100}%)"
        )

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        previous_total_value = self._total_value

        self._step_no += 1
        self._current_price_data = self._price_data.iloc[self._step_no]
        self._share_price = self._current_price_data["Adj Close"]
        next_share_price = self._price_data.iloc[self._step_no]["Adj Close"]

        if action != 6:
            # Buy or sell
            shares_to_move = self.calc_max_shares_to_buy(action)

            if action < 3:
                # Buy
                self._shares_owned_no += shares_to_move
                self._cash -= self._share_price * shares_to_move
            elif self._shares_owned_no >= shares_to_move:
                # Sell
                self._shares_owned_no -= shares_to_move
                self._cash += self._share_price * shares_to_move
            elif self._shares_owned_no > 0:
                # Sell remaining shares
                self._shares_owned_no = 0
                self._cash += self._share_price * self._shares_owned_no

        self._share_value = self._share_price * self._shares_owned_no
        self._total_value = self._cash + self._share_value

        observation = self._get_obs()
        reward = self._total_value - previous_total_value

        terminated = bool(
            self._step_no == len(self._price_data) - 1
            or self._cash < next_share_price
            or self._cash <= 0
        )

        if self.render_mode == "human":
            self.render()

        self._add_trade_to_history(
            self._step_no,
            self._cash,
            self._shares_owned_no,
            self._share_value,
            self._total_value,
            self._get_action_meaning(action),
        )

        return (
            observation,
            reward,
            terminated,
            False,
            self.trade_history,
        )

    def render(self, mode="human"):
        if mode == "human":
            print(f"Step: {self._step_no}")
            print(f"Cash: {self._cash}")
            print(f"Shares owned no: {self._shares_owned_no}")
            print(f"Share price: {self._share_price}")
            print(f"Share value: {self._share_value}")
            print(f"Total value: {self._total_value}")
            print(f"Current price data: {self._current_price_data}")
            print()
