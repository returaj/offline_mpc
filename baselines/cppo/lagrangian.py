# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of Lagrange."""

from __future__ import annotations

from collections import deque


class PIDLagrangian:
    """PID Lagrangian multiplier for constrained optimization.

    Args:
        cost_limit: the cost limit
        lagrangian_multiplier_init: the initial value of the lagrangian multiplier
        pid_kp: the proportional gain of the PID controller
        pid_ki: the integral gain of the PID controller
        pid_kd: the derivative gain of the PID controller
        pid_d_delay: the delay of the derivative term
        pid_delta_p_ema_alpha: the exponential moving average alpha of the delta_p
        pid_delta_d_ema_alpha: the exponential moving average alpha of the delta_d
        sum_norm: whether to normalize the sum of the PID output
        diff_norm: whether to normalize the difference of the PID output
        penalty_max: the maximum value of the penalty

    Attributes:
        cost_limit: the cost limit
        lagrangian_multiplier_init: the initial value of the lagrangian multiplier
        pid_kp: the proportional gain of the PID controller
        pid_ki: the integral gain of the PID controller
        pid_kd: the derivative gain of the PID controller
        pid_d_delay: the delay of the derivative term
        pid_delta_p_ema_alpha: the exponential moving average alpha of the delta_p
        pid_delta_d_ema_alpha: the exponential moving average alpha of the delta_d
        sum_norm: whether to normalize the sum of the PID output
        diff_norm: whether to normalize the difference of the PID output
        penalty_max: the maximum value of the penalty

    References:
        - Title: Responsive Safety in Reinforcement Learning by PID Lagrangian Methods
        - Authors: Adam Stooke, Joshua Achiam, Pieter Abbeel.
        - URL: `CPPOPID <https://arxiv.org/abs/2007.03964>`_
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float = 0.005,
        pid_kp: float = 0.1,
        pid_ki: float = 0.01,
        pid_kd: float = 0.01,
        pid_d_delay: int = 10,
        pid_delta_p_ema_alpha: float = 0.95,
        pid_delta_d_ema_alpha: float = 0.95,
        sum_norm: bool = True,
        diff_norm: bool = False,
        penalty_max: int = 100.0,
    ) -> None:
        """Initialize an instance of :class:`PIDLagrangian`."""
        self._pid_kp: float = pid_kp
        self._pid_ki: float = pid_ki
        self._pid_kd: float = pid_kd
        self._pid_d_delay = pid_d_delay
        self._pid_delta_p_ema_alpha: float = pid_delta_p_ema_alpha
        self._pid_delta_d_ema_alpha: float = pid_delta_d_ema_alpha
        self._penalty_max: int = penalty_max
        self._sum_norm: bool = sum_norm
        self._diff_norm: bool = diff_norm
        self._pid_i: float = lagrangian_multiplier_init
        self._cost_ds: deque[float] = deque(maxlen=self._pid_d_delay)
        self._cost_ds.append(0.0)
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0
        self._cost_limit: float = cost_limit
        self._cost_penalty: float = 0.0

    @property
    def lagrangian_multiplier(self) -> float:
        """The lagrangian multiplier."""
        return self._cost_penalty

    def update_lagrange_multiplier(self, ep_cost_avg: float) -> None:
        delta = float(ep_cost_avg - self._cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        if self._diff_norm:
            self._pid_i = max(0.0, min(1.0, self._pid_i))
        a_p = self._pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self._pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d
        self._cost_penalty = max(0.0, pid_o)
        if self._diff_norm:
            self._cost_penalty = min(1.0, self._cost_penalty)
        if not (self._diff_norm or self._sum_norm):
            self._cost_penalty = min(self._cost_penalty, self._penalty_max)
        self._cost_ds.append(self._cost_d)
