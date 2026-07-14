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

from tamarl.envs.components.metrics import compute_empirical_nash_metrics_tensor
from tamarl.envs.components.path_enumerator import get_or_compute_top_k_paths
from tamarl.envs.components.route_utils import build_routes_csr
from tamarl.envs.components.time_dependent_evaluator import TimeDependentEvaluator
from tamarl.envs.dta_bandit_env import DTABanditEnv


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
        reload_paths: bool = False,
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

        edge_eps = scenario.edge_endpoints.numpy()  # [E, 2]

        # ── Collecte des infos pour TOUS les legs ────────────────────
        self.leg_to_agent = []  # liste de (agent_idx, leg_in_agent_idx)
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
            np.stack([leg_origins, leg_dests], axis=1), axis=0, return_inverse=True
        )
        # od_indices_all_legs : utilisé UNIQUEMENT pour les métriques Nash
        # (calcul du regret par paire OD). La sélection de route utilise
        # toujours candidate_routes[od_indices, action].
        self.unique_od = unique_od
        self.od_indices_all_legs = torch.tensor(inverse_od, dtype=torch.long, device=self._device)
        self.first_edges_all_legs = torch.tensor(
            self.leg_first_edges, dtype=torch.long, device=self._device
        )

        self.num_od_pairs = len(unique_od)  # Pour les métriques uniquement

        # ── aggregation_indices : toujours zéros (B = 1) ─────────────
        # Tous les legs pointent vers le seul bloc de paramètres [0].
        self.centralized_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)

        # Free-flow times pour l'énumération des chemins
        ff_times = torch.floor(scenario.edge_static[:, 4] / timestep).numpy().astype(np.float64)

        paths_dict = get_or_compute_top_k_paths(
            scenario_dir=bandit._scenario_path,
            num_nodes=scenario.num_nodes,
            edge_endpoints=edge_eps,
            ff_times=ff_times,
            od_pairs=unique_od.astype(np.int32),
            k=top_k,
            force_recompute=reload_paths,
        )

        # ── Build candidate routes in CSR format (memory-efficient) ────
        num_unique_od = len(unique_od)
        flat_np, offsets_np, masks_np, fftt_np = build_routes_csr(
            paths_dict=paths_dict,
            unique_od=unique_od,
            top_k=top_k,
            edge_static_np=scenario.edge_static.numpy(),
        )
        self.routes_flat_csr = torch.tensor(flat_np, dtype=torch.int32, device=self._device)
        self.routes_offsets_csr = torch.tensor(offsets_np, dtype=torch.long, device=self._device)
        self.action_masks = torch.tensor(masks_np, dtype=torch.bool, device=self._device)
        self.fftt_matrix = fftt_np
        self.num_unique_od = num_unique_od

        # Check for agents/legs with no possible paths
        od_has_no_paths = ~masks_np.any(axis=1)
        inverse_od_np = self.od_indices_all_legs.cpu().numpy()
        no_path_legs = int(np.sum(od_has_no_paths[inverse_od_np]))
        if no_path_legs > 0:
            print(f"⚠️ WARNING: {no_path_legs} agents/legs have no possible routes in the network.")

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
        # self.bandit._track_events = True is no longer needed since
        # TimeDependentEvaluator now uses interval link travel times.
        self.evaluator = TimeDependentEvaluator.from_wrapper(self)

    def _get_obs(self) -> np.ndarray:
        """Observation aveugle (masques d'action) pour chaque leg."""
        # [TotalLegs, K] — indexé par OD pour obtenir les bons masques
        obs_t = self.action_masks[self.od_indices_all_legs].float()
        return obs_t.cpu().numpy()

    def _get_info(self, travel_times: np.ndarray | None = None) -> dict[str, Any]:
        """Masques + indices centralisés pour le runner."""
        # [TotalLegs, K]
        masks_t = self.action_masks[self.od_indices_all_legs]

        info: dict[str, Any] = {
            "action_mask": masks_t.cpu().numpy(),
            # aggregation_indices = zeros → tous les legs → bloc 0
            "od_indices": self.centralized_indices.cpu().numpy(),
            # Exposé pour compatibilité avec les logs
            "num_models": self.num_models,
            # od_indices réels pour les métriques Nash (pas utilisé comme mapping)
            "od_indices_for_metrics": self.od_indices_all_legs.cpu().numpy(),
        }
        if travel_times is not None:
            info.update(
                {
                    "travel_times": travel_times,
                    "mean_travel_time": float(travel_times.mean()),
                }
            )
        return info

    # ══════════════════════════════════════════════════════════════════
    #  VectorEnv API
    # ══════════════════════════════════════════════════════════════════

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return self._get_obs(), self._get_info()

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
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

        # ── CSR route lookup ────────────────────────────────────────────
        route_rows = self.od_indices_all_legs * self.K + actions_t
        route_starts = self.routes_offsets_csr[route_rows]
        route_ends = self.routes_offsets_csr[route_rows + 1]
        route_lens = (route_ends - route_starts).long()

        num_legs_np = self.bandit.scenario.num_legs.cpu()
        agent_per_leg = torch.tensor(
            [i for i, _ in self.leg_to_agent], dtype=torch.long, device=self._device
        )
        leg_total = 1 + route_lens
        agent_flat_len = torch.zeros(A, device=self._device, dtype=torch.long)
        agent_flat_len.scatter_add_(0, agent_per_leg, leg_total)
        agent_flat_len += num_legs_np.to(self._device).long() - 1

        path_offsets = torch.zeros(A + 1, device=self._device, dtype=torch.long)
        path_offsets[1:] = torch.cumsum(agent_flat_len, dim=0)
        total_flat_len = int(path_offsets[-1].item())
        paths_flat = torch.empty(total_flat_len, device=self._device, dtype=torch.int32)

        # ── Write components using chunked vectorization ──────────────
        leg_contrib_with_sep = leg_total.clone()
        first_mask = torch.zeros(self.num_envs, device=self._device, dtype=torch.bool)
        first_mask[0] = True
        if self.num_envs > 1:
            first_mask[1:] = agent_per_leg[1:] != agent_per_leg[:-1]
        leg_contrib_with_sep[~first_mask] += 1

        global_cs = torch.cumsum(leg_contrib_with_sep, dim=0)
        agent_cs_start = torch.zeros(A, device=self._device, dtype=torch.long)
        first_leg_pos = torch.nonzero(first_mask, as_tuple=True)[0]
        agent_cs_start[agent_per_leg[first_leg_pos]] = (
            global_cs[first_leg_pos] - leg_contrib_with_sep[first_leg_pos]
        )
        intra_offset = global_cs - leg_contrib_with_sep - agent_cs_start[agent_per_leg]
        leg_write_start = path_offsets[agent_per_leg] + intra_offset

        non_first = ~first_mask
        if non_first.any():
            paths_flat[leg_write_start[non_first]] = -2
            leg_write_start[non_first] += 1

        paths_flat[leg_write_start] = self.first_edges_all_legs.int()

        total_route_edges = int(route_lens.sum().item())
        if total_route_edges > 0:
            CHUNK_SIZE = 65536
            for start_idx in range(0, self.num_envs, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, self.num_envs)
                chunk_route_lens = route_lens[start_idx:end_idx]
                chunk_total_edges = int(chunk_route_lens.sum().item())
                if chunk_total_edges == 0:
                    continue

                chunk_leg_of_edge = torch.repeat_interleave(
                    torch.arange(start_idx, end_idx, device=self._device, dtype=torch.long),
                    chunk_route_lens,
                )

                chunk_cumsum_lens = torch.zeros(
                    end_idx - start_idx + 1, device=self._device, dtype=torch.long
                )
                chunk_cumsum_lens[1:] = torch.cumsum(chunk_route_lens, dim=0)

                edge_rank = (
                    torch.arange(chunk_total_edges, device=self._device, dtype=torch.long)
                    - chunk_cumsum_lens[chunk_leg_of_edge - start_idx]
                )

                src_idx = route_starts[chunk_leg_of_edge] + edge_rank
                dst_idx = leg_write_start[chunk_leg_of_edge] + 1 + edge_rank
                paths_flat[dst_idx] = self.routes_flat_csr[src_idx]

                del chunk_leg_of_edge, chunk_cumsum_lens, edge_rank, src_idx, dst_idx

        del leg_total, leg_contrib_with_sep, first_mask, global_cs, agent_cs_start
        del intra_offset, non_first, first_leg_pos, agent_per_leg
        del route_rows, route_starts, route_ends, route_lens
        import gc

        gc.collect()

        # ── Simulation bandit ─────────────────────────────────────────
        paths_flat = paths_flat.detach().contiguous()
        path_offsets = path_offsets.detach().contiguous()
        self.bandit.reset(paths_flat=paths_flat, path_offsets=path_offsets)
        _ = self.bandit.step()

        # ── Extraction des récompenses par leg ────────────────────────
        tt_matrix = self.bandit.dnl.leg_metrics[:, :, 1]

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        tt_obs = torch.zeros(self.num_envs, device=self._device)
        for idx, (i, leg_j) in enumerate(self.leg_to_agent):
            tt = float(tt_matrix[i, leg_j].item())
            rewards[idx] = -tt
            tt_obs[idx] = tt

        # ── Build valid_leg_mask: True for legs that actually departed ─
        dep_matrix = self.bandit.dnl.leg_departure_times  # [A, MaxLegs]
        dep_per_leg = torch.tensor(
            [dep_matrix[i, leg_j].item() for i, leg_j in self.leg_to_agent],
            device=self._device,
        )
        valid_leg_mask = dep_per_leg >= 0

        # ── Semi-bandit feedback ──
        # Note: semi-bandit is not implemented for the centralized wrapper.
        semi_bandit_costs = None

        # ── Packaging des sorties (filtered by valid_leg_mask) ────────
        terminated = np.ones(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)

        valid_mask_np = valid_leg_mask.cpu().numpy()
        travel_times = (-rewards).astype(np.float32)
        valid_travel_times = travel_times[valid_mask_np]

        info = self._get_info(valid_travel_times if valid_travel_times.size > 0 else travel_times)
        if semi_bandit_costs is not None:
            info["semi_bandit_feedback"] = semi_bandit_costs.cpu().numpy()
        info["_episode"] = {
            "r": rewards[valid_mask_np],
            "l": np.ones(int(valid_mask_np.sum()), dtype=np.int32),
            "t": valid_travel_times,
        }
        info["valid_leg_mask"] = valid_mask_np

        n_masked = int((~valid_leg_mask).sum().item())
        if n_masked > 0:
            info["n_masked_legs"] = n_masked

        if hasattr(self.bandit.dnl, "n_imputed_legs"):
            info["n_imputed_legs"] = int(self.bandit.dnl.n_imputed_legs)

        # ── Métriques Nash empiriques ─────────────────────────────────
        # Les métriques Nash sont calculées par paire OD réelle (num_od_pairs),
        # pas par bloc de paramètres (num_models = 1).
        if self.bandit.collect_link_tt:
            estimated_times, _ = self.evaluator.evaluate(
                dnl=self.bandit.dnl,
                departure_times=self.bandit.scenario.departure_times,
                od_indices=self.od_indices_all_legs,
            )

            path_metrics = compute_empirical_nash_metrics_tensor(
                actual_travel_times=torch.tensor(travel_times, device=self._device),
                actions=actions_t,
                estimated_times=estimated_times,
                valid_mask=valid_leg_mask,
            )
            info.update(path_metrics)

        return self._get_obs(), rewards, terminated, truncated, info

    # ══════════════════════════════════════════════════════════════════
    #  Utility
    # ══════════════════════════════════════════════════════════════════

    def get_candidate_paths_info(self) -> dict[str, Any]:
        """Résumé de la structure des chemins candidates."""
        total_edges = int(self.routes_flat_csr.shape[0])
        num_routes = self.num_unique_od * self.K
        return {
            "num_models": self.num_models,
            "num_od_pairs": self.num_od_pairs,
            "K": self.K,
            "num_vehicles": self.num_envs,
            "routes_flat_size": total_edges,
            "avg_route_len": round(total_edges / num_routes, 1) if num_routes > 0 else 0,
        }

    def close(self, **kwargs):
        """Libère les ressources."""
        pass
