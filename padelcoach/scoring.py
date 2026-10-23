"""Padel scoring engine (FIP): points 0/15/30/40, deuce with advantage or golden
point, games, sets to 6 (2 clear) with 7-point tie-break at 6-6, best of 3/5.
Tracks the serving team and changeovers (after odd games)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

PT = ["0", "15", "30", "40"]


@dataclass
class ScoreState:
    sets: List[List[int]] = field(default_factory=lambda: [[0, 0]])
    points: List[int] = field(default_factory=lambda: [0, 0])
    tiebreak: bool = False
    server: int = 0            # 0 = team A, 1 = team B
    match_over: bool = False
    sets_won: List[int] = field(default_factory=lambda: [0, 0])

    def copy(self) -> "ScoreState":
        return ScoreState([s[:] for s in self.sets], self.points[:], self.tiebreak, self.server,
                          self.match_over, self.sets_won[:])

    def games(self) -> List[int]:
        return self.sets[-1]

    def point_label(self, golden_point: bool = False) -> str:
        a, b = self.points
        if self.tiebreak:
            return f"{a}-{b}"
        if a >= 3 and b >= 3:
            if a == b:
                return "Golden point" if golden_point else "Deuce"
            return "Adv A" if a > b else "Adv B"
        return f"{PT[min(a, 3)]}-{PT[min(b, 3)]}"

    def to_dict(self, golden_point: bool = False) -> dict:
        return {"sets": [s[:] for s in self.sets], "points": self.points[:], "label": self.point_label(golden_point),
                "tiebreak": self.tiebreak, "server": "AB"[self.server], "sets_won": self.sets_won[:],
                "games_text": " | ".join(f"{s[0]}-{s[1]}" for s in self.sets)}


class PadelScorer:
    def __init__(self, golden_point: bool = False, best_of: int = 3, first_server: int = 0,
                 start: Optional[ScoreState] = None):
        self.golden_point = golden_point
        self.best_of = best_of
        self.state = start.copy() if start else ScoreState(server=first_server)
        self.games_in_match = 0

    def award(self, team: int) -> dict:
        """Award a point to team (0/1). Returns event info (game/set won, changeover)."""
        st = self.state
        info = {"game_won": None, "set_won": None, "changeover": False, "match_won": None}
        if st.match_over:
            return info
        st.points[team] += 1
        a, b = st.points
        o = 1 - team
        if st.tiebreak:
            if st.points[team] >= 7 and st.points[team] - st.points[o] >= 2:
                self._win_game(team, info)
            elif (a + b) % 2 == 1:
                st.server = 1 - st.server  # tie-break: serve changes after 1st point, then every 2
            if (a + b) % 6 == 0 and not info["game_won"]:
                info["changeover"] = True
            return info
        if self.golden_point and a >= 3 and b >= 3 and a + b == 7:
            self._win_game(team, info)  # golden point decides
        elif st.points[team] >= 4 and st.points[team] - st.points[o] >= 2:
            self._win_game(team, info)
        return info

    def _win_game(self, team: int, info: dict) -> None:
        st = self.state
        g = st.games()
        g[team] += 1
        info["game_won"] = "AB"[team]
        st.points = [0, 0]
        was_tb = st.tiebreak
        st.tiebreak = False
        self.games_in_match += 1
        st.server = 1 - st.server
        a, b = g
        set_done = (max(a, b) >= 6 and abs(a - b) >= 2) or was_tb or max(a, b) == 7
        if set_done:
            st.sets_won[team] += 1
            info["set_won"] = "AB"[team]
            if st.sets_won[team] > self.best_of // 2:
                st.match_over = True
                info["match_won"] = "AB"[team]
            else:
                st.sets.append([0, 0])
        elif a == 6 and b == 6:
            st.tiebreak = True
        total = sum(g) if not set_done else 1
        info["changeover"] = total % 2 == 1
