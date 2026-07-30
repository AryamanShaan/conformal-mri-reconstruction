import contextlib
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch


# -------------------------------------------#
# ----------------- temp seed -------------- #
# -------------------------------------------#
@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]]):
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


# -------------------------------------------#
# ---------------- mask func --------------- #
# -------------------------------------------#
class MaskFunc:

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):
        if len(center_fractions) != len(accelerations) and not allow_any_combination:
            raise ValueError(
                "Number of center fractions should match number of accelerations "
                "if allow_any_combination is False."
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:

        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_low_frequencies = self.sample_mask(
                shape, offset
            )

        return torch.max(center_mask, accel_mask), num_low_frequencies

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:

        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)

        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, acceleration_mask, num_low_frequencies

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs
        return mask

    def choose_acceleration(self):
        if self.allow_any_combination:
            return self.rng.choice(self.center_fractions), self.rng.choice(
                self.accelerations
            )
        else:
            choice = self.rng.randint(len(self.center_fractions))
            return self.center_fractions[choice], self.accelerations[choice]


# -------------------------------------------#
# ------------ equispaced mask ------------- #
# -------------------------------------------#
class EquiSpacedMaskFunc(MaskFunc):

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:

        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask


# -------------------------------------------#
# ------- equispaced fraction mask --------- #
# -------------------------------------------#
class EquispacedMaskFractionFunc(MaskFunc):

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        adjusted_accel = (acceleration * (num_low_frequencies - num_cols)) / (
            num_low_frequencies * acceleration - num_cols
        )
        if offset is None:
            offset = self.rng.randint(0, high=round(adjusted_accel))

        mask = np.zeros(num_cols)
        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = 1.0

        return mask


# -------------------------------------------#
# ------ random variable-density mask ------ #
# -------------------------------------------#
class RandomVariableDensityMaskFunc(MaskFunc):
    """
    Random variable-density sampling mask compatible with the set-1 MaskFunc API.

    For each call:
        1. Sample k_fraction uniformly from [min_k_fraction, max_k_fraction].
        2. Set center_fraction = k_fraction / 2.
        3. Fully sample the center block.
        4. Sample the remaining lines without replacement using a variable-density
           prior that favors locations near the k-space center.

    Public API remains compatible with set 1:
        mask, num_low_frequencies = mask_func(shape, offset=None, seed=None)
    """

    def __init__(
            
        self,
        # center_fractions: Sequence[float],
        # accelerations: Sequence[int],
        min_k_fraction: float,
        max_k_fraction: float,
        # allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):
        # super().__init__(
        #     center_fractions=center_fractions,
        #     accelerations=accelerations,
        #     allow_any_combination=allow_any_combination,
        #     seed=seed,
        # )

        if min_k_fraction < 0 or max_k_fraction <= 0:
            raise ValueError("min_k_fraction and max_k_fraction must be positive.")
        if min_k_fraction > max_k_fraction:
            raise ValueError("min_k_fraction must be <= max_k_fraction.")
        if max_k_fraction > 1:
            raise ValueError("max_k_fraction must be <= 1.")
        if min_k_fraction > 1:
            raise ValueError("min_k_fraction must be <= 1.")

        self.min_k_fraction = min_k_fraction
        self.max_k_fraction = max_k_fraction

        self.rng = np.random.RandomState(seed)

    def _get_center_bounds(self, num_cols: int, num_low_frequencies: int) -> tuple[int, int]:
        pad = (num_cols - num_low_frequencies + 1) // 2
        return pad, pad + num_low_frequencies

    # def _get_prior(self, candidate_indices: np.ndarray, num_cols: int) -> np.ndarray:
    #     """
    #     Center-biased prior over candidate outer indices.
    #     """
    #     if len(candidate_indices) == 0:
    #         return np.array([], dtype=np.float64)

    #     center = (num_cols - 1) / 2.0
    #     distances = np.abs(candidate_indices.astype(np.float64) - center)

    #     # Larger weight near center, smaller toward edges.
    #     weights = 1.0 / (distances + 1.0)
    #     weights /= weights.sum()
    #     return weights

    def _get_prior(self, remaining_indices: Sequence[int]) -> np.ndarray:

        n_cols = len(remaining_indices)

        if n_cols % 2 == 0:
            dist = np.arange(1, n_cols // 2 + 1)
            dist = np.r_[dist, dist[::-1]]
        else:
            dist = np.arange(1, n_cols // 2 + 2)
            dist = np.r_[dist, dist[::-1][:-1]]
        return dist / dist.sum()

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Override MaskFunc.sample_mask because here center_fraction is not chosen
        from self.center_fractions. Instead it is derived from a randomly chosen
        k_fraction for each call.
        """
        num_cols = shape[-2]

        # Sample total sampling fraction uniformly.
        k_fraction = self.rng.uniform(self.min_k_fraction, self.max_k_fraction)

        # Center fraction is half of chosen k_fraction.
        center_fraction = k_fraction / 2.0

        num_low_frequencies = round(num_cols * center_fraction)

        center_mask_1d = self.calculate_center_mask(shape, num_low_frequencies)
        accel_mask_1d = self.calculate_acceleration_mask(
            num_cols=num_cols,
            num_low_frequencies=num_low_frequencies,
            k_fraction=k_fraction,
        )

        center_mask = self.reshape_mask(center_mask_1d, shape)
        accel_mask = self.reshape_mask(accel_mask_1d, shape)

        return center_mask, accel_mask, num_low_frequencies

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        num_low_frequencies: int,
        k_fraction: Optional[float] = None,
    ) -> np.ndarray:
        """
        Build outer variable-density mask.
        """
        if k_fraction is None:
            raise ValueError("k_fraction must be provided for RandomVariableDensityMaskFunc.")

        mask = np.zeros(num_cols, dtype=np.float32)

        # Desired total number of sampled lines.
        target_total = round(num_cols * k_fraction)

        # Need this many additional samples outside the center.
        num_outer_samples = max(0, target_total - num_low_frequencies)
        if num_outer_samples == 0:
            return mask

        center_start, center_end = self._get_center_bounds(num_cols, num_low_frequencies)
        all_indices = np.arange(num_cols)
        candidate_indices = np.concatenate(
            [all_indices[:center_start], all_indices[center_end:]]
        )

        if len(candidate_indices) == 0:
            return mask

        num_outer_samples = min(num_outer_samples, len(candidate_indices))
        sampling_probs = self._get_prior(candidate_indices)

        chosen = self.rng.choice(
            candidate_indices,
            size=num_outer_samples,
            replace=False,
            p=sampling_probs,
        )

        mask[chosen] = 1.0
        return mask


# -------------------------------------------#
# -------- create mask for mask type ------- #
# -------------------------------------------#
def create_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
    min_k_fraction: Optional[float] = None,
    max_k_fraction: Optional[float] = None,
) -> MaskFunc:

    if mask_type_str == "equispaced":
        return EquiSpacedMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced_fraction":
        return EquispacedMaskFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "random_vds":
        if min_k_fraction is None or max_k_fraction is None:
            raise ValueError(
                "random_vds requires min_k_fraction and max_k_fraction."
            )
        return RandomVariableDensityMaskFunc(min_k_fraction, max_k_fraction)
    else:
        raise ValueError(f"{mask_type_str} not supported")
