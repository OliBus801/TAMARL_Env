"""CentralizedLevelWrapper — gymnasium.vector.VectorEnv bridge for DTABanditEnv.

Mode d'agrégation centralisé : un seul modèle de paramètres (B = 1) est
partagé par l'ensemble des N véhicules/legs. Tous les agents prennent des
décisions individuelles, mais le signal d'apprentissage de tous les N
véhicules est agrégé vers ce seul bloc de paramètres.

Ce wrapper expose :
  - ``num_models = 1`` : l'agent sera initialisé avec B = 1.
  - ``centralized_indices`` in info : tenseur de zéros [TotalLegs],
    jouant le rôle de ``aggregation_indices`` dans train_bandit.

Grâce à l'architecture agnostique, tous les agents (EpsilonGreedy, Exp3,
UCB, TS, etc.) fonctionnent sans modification : ils reçoivent simplement
aggregation_indices = zeros(N) et mettent à jour leurs poids [1, K].

Flow:
    1. __init__  → chargement scénario, énumération top-k chemins par OD,
                   construction candidate_routes [Num_Unique_OD, K, MaxLen].
    2. reset()   → retourne obs + info (incl. od_indices pour les métriques).
    3. step(actions)
         a. Assemblage du tenseur multi-leg [A, MaxPathLen].
         b. Run: ``bandit.reset(paths); bandit.step()``
         c. Retour: obs, rewards, terminated, truncated, info.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from gymnasium import spaces

from tamarl.envs.dta_bandit_env import DTABanditEnv
from tamarl.envs.components.path_enumerator import get_or_compute_top_k_paths
from tamarl.envs.components.metrics import compute_empirical_nash_metrics_tensor
from tamarl.envs.components.time_dependent_evaluator import TimeDependentEvaluator


class CentralizedLevelWrapper:
    """Wrapper centralisé : B = 1, tous les véhicules partagent un seul modèle.

    Structurellement identique à ODLevelWrapper, mais expose
    ``aggregation_indices = zeros(N)`` afin que le runner ``train_bandit``
    instruite l'agent d'agréger toute l'expérience des N véhicules vers
    un seul bloc de paramètres [1, K].

    Ce mode est l'équivalent bandit d'un apprentissage par renforcement
    centralisé : le modèle apprend la « meilleure route en moyenne » sur
    l'ensemble du réseau, indépendamment des paires OD.

    Attributes:
        bandit:               Moteur DTABanditEnv sous-jacent.
        candidate_routes:     [Num_Unique_OD, K, MaxRouteLen] padded with -1.
        od_indices_all_legs:  [TotalLegs] mapping leg → indice OD unique.
        K:                    Nombre de routes candidates K.
        num_envs:             Nombre total de legs.
        num_models:           Toujours 1 (mode centralisé).
    """

    metadata = {"render_modes": [], "autoreset_mode": 0}

    def __init__(
        self,
        bandit: DTABanditEnv,
        top_k: int = 3,
        feedback_type: str = "full",
    ):
        self.bandit = bandit
        self.K = top_k
        self._device = bandit._device
        self.feedback_type = feedback_type

        # Exposé pour compatibilité avec le runner (analogue à num_od_pairs)
        self.num_models = 1

        scenario = self.bandit.scenario
        A = self.bandit.num_agents
        timestep = bandit._timestep

        edge_eps = scenario.edge_endpoints.numpy()   # [E, 2]
        edge_static = scenario.edge_static.numpy()

        # ── Collecte des infos pour TOUS les legs ────────────────────
        self.leg_to_agent = []       # liste de (agent_idx, leg_in_agent_idx)
        leg_origins = []
        leg_dests = []
        self.leg_first_edges = []

        num_legs_np = scenario.num_legs.numpy()
        fe_np = scenario.first_edges.numpy()
        dest_np = scenario.destinations.numpy()

        for i in range(A):
            for leg in range(num_legs_np[i]):
                fe = int(fe_np[i, leg])
                dest = int(dest_np[i, leg])
                if fe >= 0:
                    orig = int(edge_eps[fe, 1])
                    leg_origins.append(orig)
                    leg_dests.append(dest)
                    self.leg_first_edges.append(fe)
                    self.leg_to_agent.append((i, leg))

        self.num_envs = len(self.leg_to_agent)

        # ── Énumération des top-K chemins pour toutes les OD uniques ──
        unique_od, inverse_od = np.unique(
            np.stack([leg_origins, leg_dests], axis=1),
            axis=0, return_inverse=True
        )
        # od_indices_all_legs : utilisé UNIQUEMENT pour les métriques Nash
        # (calcul du regret par paire OD). La sélection de route utilise
        # toujours candidate_routes[od_indices, action].
        self.od_indices_all_legs = torch.tensor(
            inverse_od, dtype=torch.long, device=self._device
        )
        self.first_edges_all_legs = torch.tensor(
            self.leg_first_edges, dtype=torch.long, device=self._device
        )

        self.num_od_pairs = len(unique_od)  # Pour les métriques uniquement

        # ── aggregation_indices : toujours zéros (B = 1) ─────────────
        # Tous les legs pointent vers le seul bloc de paramètres [0].
        self.centralized_indices = torch.zeros(
            self.num_envs, dtype=torch.long, device=self._device
        )

        # Free-flow times pour l'énumération des chemins
        ff_times = torch.floor(
            scenario.edge_static[:, 4] / timestep
        ).numpy().astype(np.float64)

        paths_dict = get_or_compute_top_k_paths(
            scenario_dir=bandit._scenario_path,
            num_nodes=scenario.num_nodes,
            edge_endpoints=edge_eps,
            ff_times=ff_times,
            od_pairs=unique_od.astype(np.int32),
            k=top_k,
        )

        # ── Construction de candidate_routes [Num_Unique_OD, K, MaxLen] ─
        max_route_len = 0
        for paths_list in paths_dict.values():
            for p in paths_list:
                max_route_len = max(max_route_len, len(p))
        max_route_len = max(max_route_len, 1)

        num_unique_od = len(unique_od)
        cand_np = np.full((num_unique_od, top_k, max_route_len), -1, dtype=np.int32)
        masks_np = np.zeros((num_unique_od, top_k), dtype=bool)

        for od_idx in range(num_unique_od):
            od_key = (int(unique_od[od_idx, 0]), int(unique_od[od_idx, 1]))
            paths_list = paths_dict.get(od_key, [])
            if not paths_list:
                continue

            for k_idx in range(top_k):
                p_idx = min(k_idx, len(paths_list) - 1)
                path = paths_list[p_idx]
                for e_idx, edge_id in enumerate(path):
                    cand_np[od_idx, k_idx, e_idx] = edge_id
                masks_np[od_idx, k_idx] = (k_idx < len(paths_list))

        self.candidate_routes = torch.tensor(
            cand_np, dtype=torch.long, device=self._device
        )  # [Num_Unique_OD, K, MaxRouteLen]

        self.action_masks = torch.tensor(
            masks_np, dtype=torch.bool, device=self._device
        )  # [Num_Unique_OD, K]

        # ── Matrice FFTT pour le calcul du regret Nash ─────────────────
        self.fftt_matrix = np.zeros((num_unique_od, top_k), dtype=np.float32)
        edge_static_np = scenario.edge_static.numpy()
        for od_idx in range(num_unique_od):
            for k_idx in range(top_k):
                if masks_np[od_idx, k_idx]:
                    path = cand_np[od_idx, k_idx]
                    valid_path = path[path != -1]
                    path_fftt = edge_static_np[valid_path, 4].sum()
                    self.fftt_matrix[od_idx, k_idx] = path_fftt
                else:
                    self.fftt_matrix[od_idx, k_idx] = np.inf

        # ── Longueur max des chemins multi-leg ─────────────────────────
        max_total = 0
        for i in range(A):
            n = num_legs_np[i]
            total = n * (1 + max_route_len) + (n - 1)
            max_total = max(max_total, total)
        self._max_path_len = int(max_total)

        # ── Gymnasium VectorEnv setup ─────────────────────────────────
        self.single_observation_space = spaces.Box(
            low=0, high=np.inf, shape=(top_k,), dtype=np.float32
        )
        self.single_action_space = spaces.Discrete(top_k)

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.num_envs, top_k), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([top_k] * self.num_envs)

        self.spec = None
        self.render_mode = None
        self.closed = False

        # Enforce event tracking for TimeDependentEvaluator (needed for Nash metrics)
        self.bandit._track_events = True
        self.evaluator = TimeDependentEvaluator.from_wrapper(self)

    def _get_obs(self) -> np.ndarray:
        """Observation aveugle (masques d'action) pour chaque leg."""
        # [TotalLegs, K] — indexé par OD pour obtenir les bons masques
        obs_t = self.action_masks[self.od_indices_all_legs].float()
        return obs_t.cpu().numpy()

    def _get_info(self, travel_times: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Masques + indices centralisés pour le runner."""
        # [TotalLegs, K]
        masks_t = self.action_masks[self.od_indices_all_legs]

        info: Dict[str, Any] = {
            "action_mask": masks_t.cpu().numpy(),
            # aggregation_indices = zeros → tous les legs → bloc 0
            "od_indices": self.centralized_indices.cpu().numpy(),
            # Exposé pour compatibilité avec les logs
            "num_models": self.num_models,
            # od_indices réels pour les métriques Nash (pas utilisé comme mapping)
            "od_indices_for_metrics": self.od_indices_all_legs.cpu().numpy(),
        }
        if travel_times is not None:
            info.update({
                "travel_times": travel_times,
                "mean_travel_time": float(travel_times.mean()),
            })
        return info

    # ══════════════════════════════════════════════════════════════════
    #  VectorEnv API
    # ══════════════════════════════════════════════════════════════════

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._get_obs(), self._get_info()

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Exécute une simulation DTA complète.

        Args:
            actions: [TotalLegs] int array. Route sélectionnée par chaque leg.
        """
        # ── Blindage de l'input ──────────────────────────────────────
        if isinstance(actions, torch.Tensor):
            actions_t = actions.detach().contiguous().to(self._device, dtype=torch.long)
        else:
            actions_t = torch.tensor(actions, dtype=torch.long, device=self._device)
        
        A = self.bandit.num_agents

        # ── Routes pour chaque leg [TotalLegs, MaxRouteLen] ──────────
        selected_routes = self.candidate_routes[self.od_indices_all_legs, actions_t]

        # ── Assemblage du tenseur multi-leg [A, MaxPathLen] ───────────
        paths = torch.full(
            (A, self._max_path_len), -1,
            dtype=torch.long, device=self._device
        )

        leg_ptr = 0
        for i in range(A):
            n_legs = self.bandit.scenario.num_legs[i].item()
            ptr = 0
            for leg_j in range(n_legs):
                if leg_j > 0:
                    paths[i, ptr] = -2  # séparateur de leg
                    ptr += 1

                paths[i, ptr] = self.first_edges_all_legs[leg_ptr]
                ptr += 1

                route = selected_routes[leg_ptr]
                valid_edges = route[route != -1]
                L = valid_edges.size(0)
                paths[i, ptr: ptr + L] = valid_edges
                ptr += L

                leg_ptr += 1

        # ── Simulation bandit ─────────────────────────────────────────
        # Blindage du tenseur paths avant passage au simulateur
        paths = paths.detach().contiguous()
        self.bandit.reset(paths)
        _ = self.bandit.step()

        # ── Extraction des récompenses par leg ────────────────────────
        tt_matrix = self.bandit.dnl.leg_metrics[:, :, 1]

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        tt_obs = torch.zeros(self.num_envs, device=self._device)
        for idx, (i, leg_j) in enumerate(self.leg_to_agent):
            tt = float(tt_matrix[i, leg_j].item())
            rewards[idx] = -tt
            tt_obs[idx] = tt

        # ── Semi-bandit feedback ──────────────────────────────────────
        semi_bandit_costs = None
        if self.feedback_type == "semi":
            dynamic_tt = self.bandit.dnl.get_dynamic_link_travel_times()
            if dynamic_tt is not None:
                edge_tt = dynamic_tt.mean(dim=0)
            else:
                edge_tt = self.bandit.dnl.edge_static[:, 4]

            safe_routes = torch.where(
                selected_routes >= 0, selected_routes, torch.zeros_like(selected_routes)
            )
            semi_bandit_costs = edge_tt[safe_routes]
            semi_bandit_costs = torch.where(
                selected_routes >= 0, semi_bandit_costs, torch.zeros_like(semi_bandit_costs)
            )

        # ── Packaging des sorties ─────────────────────────────────────
        terminated = np.ones(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)

        travel_times = (-rewards).astype(np.float32)
        info = self._get_info(travel_times)
        if semi_bandit_costs is not None:
            info["semi_bandit_feedback"] = semi_bandit_costs.cpu().numpy()
        info["_episode"] = {
            "r": rewards,
            "l": np.ones(self.num_envs, dtype=np.int32),
            "t": travel_times,
        }

        # ── Métriques Nash empiriques ─────────────────────────────────
        # Les métriques Nash sont calculées par paire OD réelle (num_od_pairs),
        # pas par bloc de paramètres (num_models = 1).
        estimated_times, _ = self.evaluator.evaluate(
            dnl=self.bandit.dnl,
            departure_times=self.bandit.scenario.departure_times,
            od_indices=self.od_indices_all_legs,
        )
        
        path_metrics = compute_empirical_nash_metrics_tensor(
            actual_travel_times=torch.tensor(travel_times, device=self._device),
            actions=actions_t,
            estimated_times=estimated_times
        )
        info.update(path_metrics)

        return self._get_obs(), rewards, terminated, truncated, info

    # ══════════════════════════════════════════════════════════════════
    #  Utility
    # ══════════════════════════════════════════════════════════════════

    def get_candidate_paths_info(self) -> Dict[str, Any]:
        """Résumé de la structure des chemins candidates."""
        return {
            "num_models": self.num_models,
            "num_od_pairs": self.num_od_pairs,
            "K": self.K,
            "max_path_len": self._max_path_len,
            "num_vehicles": self.num_envs,
            "candidate_routes_shape": list(self.candidate_routes.shape),
        }

    def close(self, **kwargs):
        """Libère les ressources."""
        pass
