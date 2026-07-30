import numpy as np
import torch
import contextlib
from typing import Optional, Sequence, Tuple, Union
import warnings

@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]]):
    """A context manager for temporarily adjusting the random seed."""
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


class MaskFuncIncremental:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    When called, ``MaskFunc`` uses internal functions create mask by 1)
    creating a mask for the k-space center, 2) create a mask outside of the
    k-space center, and 3) combining them into a total mask. The internals are
    handled by ``sample_mask``, which calls ``calculate_center_mask`` for (1)
    and ``calculate_acceleration_mask`` for (2). The combination is executed
    in the ``MaskFunc`` ``__call__`` function.

    If you would like to implement a new mask, simply subclass ``MaskFunc``
    and overwrite the ``sample_mask`` logic. See examples in ``RandomMaskFunc``
    and ``EquispacedMaskFunc``.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: Seed for starting the internal random number generator of the
                ``MaskFuncIncremental``.
        """
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.
            **kwargs: Additional subclass-specific arguments forwarded to
                ``sample_mask`` (e.g. ``previous_mask``, ``target_k_fraction``
                for ``IncrementalVariableDensityMaskFunc``).

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_low_frequencies = self.sample_mask(
                shape, offset=offset, **kwargs
            )

        # combine masks together
        return torch.max(center_mask, accel_mask), num_low_frequencies

    
    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        Subclasses must implement this. It should return 1) the center mask,
        2) the acceleration mask, and 3) the integer count of low frequency
        samples. ``__call__`` combines the two masks.
        """
        raise NotImplementedError

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(self, num_cols: int, num_low_frequencies: int, k_fraction: float) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking (for equispaced masks).
            num_low_frequencies: Integer count of low-frequency lines sampled.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        """
        Build center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.

        Returns:
            A mask for hte low spatial frequencies of k-space.
        """
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs

        return mask




class IncrementalVariableDensityMaskFunc(MaskFuncIncremental):

    # Removed init here
    # def __init__(
            
    #     self,
    #     # center_fractions: Sequence[float],
    #     # accelerations: Sequence[int],
    #     min_k_fraction: float,
    #     max_k_fraction: float,
    #     # allow_any_combination: bool = False,
    #     seed: Optional[int] = None,
    # ):
    #     # super().__init__(
    #     #     center_fractions=center_fractions,
    #     #     accelerations=accelerations,
    #     #     allow_any_combination=allow_any_combination,
    #     #     seed=seed,
    #     # )

    #     if min_k_fraction < 0 or max_k_fraction <= 0:
    #         raise ValueError("min_k_fraction and max_k_fraction must be positive.")
    #     if min_k_fraction > max_k_fraction:
    #         raise ValueError("min_k_fraction must be <= max_k_fraction.")
    #     if max_k_fraction > 1:
    #         raise ValueError("max_k_fraction must be <= 1.")
    #     if min_k_fraction > 1:
    #         raise ValueError("min_k_fraction must be <= 1.")

    #     self.min_k_fraction = min_k_fraction
    #     self.max_k_fraction = max_k_fraction

    #     self.rng = np.random.RandomState(seed)


    def _extract_previous_columns(
        self,
        previous_mask: Union[np.ndarray, torch.Tensor],
        num_cols: int,
    ) -> np.ndarray:
        """
        Reduce a previous mask to a 1-D binary array over columns.

        The previous mask is expected to have the same shape as the masks
        produced by ``reshape_mask``: singleton in every dimension except the
        column axis (``shape[-2]``).

        Args:
            previous_mask: Mask from a previous sampling round.
            num_cols: Number of k-space columns.

        Returns:
            A float32 array of length ``num_cols``, 1 where the column was
            previously sampled.
        """
        if isinstance(previous_mask, torch.Tensor):
            arr = previous_mask.detach().cpu().numpy()
        else:
            arr = np.asarray(previous_mask)

        arr = np.squeeze(arr)

        if arr.ndim != 1:
            raise ValueError(
                f"previous_mask must reduce to 1-D along the column axis; got "
                f"shape {tuple(np.shape(previous_mask))} -> {arr.shape}."
            )
        if arr.shape[0] != num_cols:
            raise ValueError(
                f"previous_mask has {arr.shape[0]} columns, expected {num_cols}."
            )

        return (arr > 0).astype(np.float32)

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


    def _get_prior_2(self, candidate_indices: np.ndarray, num_cols: int) -> np.ndarray:
        if len(candidate_indices) == 0:
            return np.array([], dtype=np.float64)

        idx = np.asarray(candidate_indices, dtype=np.float64)

        if idx.min() < 0 or idx.max() > num_cols - 1:
            raise ValueError(
                f"candidate_indices must lie in [0, {num_cols - 1}], "
                f"got range [{idx.min():.0f}, {idx.max():.0f}]."
            )

        weights = np.minimum(idx + 1.0, num_cols - idx)
        return weights / weights.sum()


    # def sample_mask(
    #     self,
    #     shape: Sequence[int],
    #     offset: Optional[int],
    # ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    #     """
    #     Override MaskFunc.sample_mask because here center_fraction is not chosen
    #     from self.center_fractions. Instead it is derived from a randomly chosen
    #     k_fraction for each call.
    #     """
    #     num_cols = shape[-2]

    #     # Sample total sampling fraction uniformly.
    #     k_fraction = self.rng.uniform(self.min_k_fraction, self.max_k_fraction)

    #     # Center fraction is half of chosen k_fraction.
    #     center_fraction = k_fraction / 2.0

    #     num_low_frequencies = round(num_cols * center_fraction)

    #     center_mask_1d = self.calculate_center_mask(shape, num_low_frequencies)
    #     accel_mask_1d = self.calculate_acceleration_mask(
    #         num_cols=num_cols,
    #         num_low_frequencies=num_low_frequencies,
    #         k_fraction=k_fraction,
    #     )

    #     center_mask = self.reshape_mask(center_mask_1d, shape)
    #     accel_mask = self.reshape_mask(accel_mask_1d, shape)

    #     return center_mask, accel_mask, num_low_frequencies


    def sample_mask(
        self,
        shape: Sequence[int],
        target_k_fraction: float,
        offset: Optional[int] = None,
        previous_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
 
        num_cols = shape[-2]

        k_fraction = target_k_fraction

        if not 0.0 < k_fraction <= 1.0:
            raise ValueError(f"k_fraction must be in (0, 1], got {k_fraction}.")


        # Center block is half the sampling budget; it grows with k_fraction.
        # center_fraction = k_fraction / 2.0
        # num_low_frequencies = round(num_cols * center_fraction)
        target_total = round(num_cols * k_fraction)
        num_low_frequencies = target_total // 2

        center_mask_1d = self.calculate_center_mask(shape, num_low_frequencies)

        if previous_mask is None:
            accel_mask_1d = self.calculate_acceleration_mask(
                num_cols=num_cols,
                num_low_frequencies=num_low_frequencies,
                k_fraction=k_fraction,
            )
        else:
            previous_mask_columns = self._extract_previous_columns(
                previous_mask, num_cols
            )

            # Feasibility check: we can add columns but never remove them.
            num_prev = int(previous_mask_columns.sum())
            # target_total = round(num_cols * k_fraction)
            if num_prev > target_total:
                warnings.warn(
                    f"previous_mask already has {num_prev} sampled columns, which "
                    f"exceeds the target of {target_total} "
                    f"(k_fraction={k_fraction:.4f}, num_cols={num_cols}). Columns "
                    f"cannot be un-sampled, so the returned mask will overshoot "
                    f"the target k_fraction.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            accel_mask_1d = self.calculate_acceleration_mask_with_prev_mask(
                num_cols=num_cols,
                num_low_frequencies=num_low_frequencies,
                k_fraction=k_fraction,
                previous_mask_columns=previous_mask_columns,
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


    def calculate_acceleration_mask_with_prev_mask(
        self,
        num_cols: int,
        num_low_frequencies: int,
        k_fraction: float,
        previous_mask_columns: np.ndarray,
    ) -> np.ndarray:
        """
        Build the outer variable-density mask, preserving all columns sampled
        in a previous mask and topping up to the target ``k_fraction``.

        Args:
            num_cols: Number of k-space columns.
            num_low_frequencies: Number of fully sampled center lines.
            k_fraction: Target total sampling fraction.
            previous_mask_columns: 1-D binary array of previously sampled columns.

        Returns:
            A mask for the high spatial frequencies of k-space, including the
            previously sampled outer lines.
        """
        mask = np.zeros(num_cols, dtype=np.float32)

        center_start, center_end = self._get_center_bounds(
            num_cols, num_low_frequencies
        )
        all_indices = np.arange(num_cols)
        prev_indices = np.flatnonzero(previous_mask_columns > 0)

        # Previously sampled lines outside the NEW center are forced keeps:
        # write them in now, and exclude them from the candidate pool.
        # Previously sampled lines INSIDE the new center are already covered by
        # calculate_center_mask, so they need no handling here.
        in_center = (prev_indices >= center_start) & (prev_indices < center_end)
        prev_outer_indices = prev_indices[~in_center]
        mask[prev_outer_indices] = 1.0

        # Budget: target total, minus the center block, minus the forced keeps.
        target_total = round(num_cols * k_fraction)
        num_new_samples = (
            target_total - num_low_frequencies - len(prev_outer_indices)
        )

        if num_new_samples < 0:
            warnings.warn(
                f"Cannot reach target of {target_total} columns: the center "
                f"({num_low_frequencies}) plus retained previous columns "
                f"({len(prev_outer_indices)}) already total "
                f"{num_low_frequencies + len(prev_outer_indices)}. Columns cannot "
                f"be un-sampled, so the mask overshoots the target.",
                RuntimeWarning, stacklevel=3,
            )
        if num_new_samples <= 0:
            return mask

        outer_indices = np.concatenate(
            [all_indices[:center_start], all_indices[center_end:]]
        )
        candidate_indices = np.setdiff1d(outer_indices, prev_outer_indices)

        if len(candidate_indices) == 0:
            return mask

        num_new_samples = min(num_new_samples, len(candidate_indices))
        sampling_probs = self._get_prior_2(candidate_indices, num_cols)

        chosen = self.rng.choice(
            candidate_indices,
            size=num_new_samples,
            replace=False,
            p=sampling_probs,
        )
        mask[chosen] = 1.0

        return mask


if __name__ == "__main__":
    # Example usage
    mask_func = IncrementalVariableDensityMaskFunc(seed=0)
    shape = (1, 320, 320, 2)

    m1, _ = mask_func(shape, target_k_fraction=0.2, seed=42)
    m2, _ = mask_func(shape, target_k_fraction=0.4, previous_mask=m1, seed=43)

    assert torch.all(m2 >= m1), "nesting violated"
    assert int(m1.sum()) == round(320 * 0.2)
    assert int(m2.sum()) == round(320 * 0.4)