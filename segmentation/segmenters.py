
"""    
This module contains classes for segmentation tasks. The module provides classes for defining score models, segmenters, and optimizers for segmentation tasks. 

Classes:
    Segmenter: A class representing a segmenter for segmentation tasks.
    Loss: Abstract base class for defining loss functions used in segmentation.
    F1Loss: A class that calculates the F1 loss for a given set of scores and ground truth break points.
    ScoreModel: Abstract base class for score models used in segmentation.
    WaveletDecompositionModel: A class representing a wavelet decomposition model for segmentation.
    Optimizer: Abstract base class for optimizers used in segmentation.
    BayesianOptimizer: A class that performs Bayesian optimization to find the optimal weights for a segmenter.
    MangoOptimizer: A class that optimizes the weights of a segmenter using the Mango optimization algorithm.
    SupervisedDomain: A class representing a supervised domain for segmentation.

Segmenter:
    A class representing a segmenter for segmentation tasks. The segmenter is used to perform segmentation on a datastream using a score model. The segmenter can be compiled with a loss function and an optimizer to find the optimal weights for the score model. 

Loss:
    Abstract base class for defining loss functions used in segmentation.

F1Loss:
    A class that calculates the F1 loss for a given set of scores and ground truth break points. The F1 loss is defined as 1-F1, where F1 is the maximum F1 score that can be achieved by varying the threshold value.

ScoreModel:
    Abtract base class for score models used in segmentation. The score model is used to calculate the score for a segmentation task. In particular, the method get_profile() return a score profile, that has the same length of the datastream to be segmented. A peak in the score profile indicates a change point in the datastream.

 WaveletDecompositionModel:
    A class representing a wavelet decomposition model for segmentation. It performs wavelet decomposition on the input samples using the specified level and wavelet type (db3). For each subband obtained from the decomposition, it calculates a profile using the NormalDivergenceProfile, resamples it, applies a power transform, and stores it. It then computes an average score profile across all subbands and sets this as both the average and the score profile of the model. 

Optimizer:
    Abstract base class for optimizers used in segmentation. The optimizer is devoted to find the optimal weights for a segmenter.

BayesianOptimizer:
    A class that performs Bayesian optimization to find the optimal weights for a segmenter.

MangoOptimizer:
    The MangoOptimizer class is used to optimize the weights of a segmenter using the Mango optimization algorithm.

SupervisedDomain:
    Represents a supervised domain for segmentation. This class is designed to manage a domain of supervised intervals for segmentation tasks. It encapsulates the concept of a domain that is partially labeled or supervised, where certain ranges (intervals) of the domain are marked as supervised regions. This class provides a structured way to handle these intervals, check for membership, add new intervals, resolve overlaps between them, and generate a boolean array indicating the supervised samples.

"""

from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np
import pywt
from scipy.signal import savgol_filter
from scipy.signal import peak_prominences, find_peaks
from segmentation.profiles import NormalDivergenceProfile, ProminenceProfile, AverageScoreProfile
from segmentation.profiles import PowerTransformProfile, ResampleProfile
from mango.tuner import Tuner
from scipy.stats import uniform # type: ignore
from segmentation.utils import MVMD
import time
import pickle
import json




class Loss(ABC):
    """
    Abstract base class for defining loss functions used in segmentation.
    """

    @abstractmethod
    def __call__(self, score: np.array, gt_break_points: List[int]) -> float:
        """
        Calculate the loss value given the predicted score and ground truth break points.

        Args:
            score (np.array): The predicted score.
            gt_break_points (List[int]): The ground truth break points.

        Returns:
            float: The calculated loss value.
        """
        pass

    @abstractmethod
    def get_optimal_threshold(self, score: np.array, gt_break_points: List[int]) -> float:
        """
        Calculate the optimal threshold value given the predicted score and ground truth break points.

        Args:
            score (np.array): The predicted score.
            gt_break_points (List[int]): The ground truth break points.

        Returns:
            float: The calculated optimal threshold value.
        """
        pass

    @abstractmethod
    def get_minimum_value(self) -> float:
        """
        Get the minimum possible value for the loss function.

        Returns:
            float: The minimum value for the loss function.
        """
        pass

class F1Loss(Loss):
    """
    F1Loss is a class that calculates the F1 loss for a given set of scores and ground truth break points. 
    The F1 loss is defined as 1-F1, where F1 is the maximum F1 score that can be achieved by varying the threshold value.
    """

    def __init__(self, threshold: float, peak_tolerance: int = 100):
        """
        Initializes the F1Loss object.

        Args:
            peak_tolerance (int): The tolerance value for matching peaks in the ground truth break points.
                                  Defaults to 100.
        """
        self.peak_tolerance = peak_tolerance
        self.threshold = threshold

    def _compute_best_f1(self, scores: np.array, gt_break_points: List[int]) -> Tuple[float, float]:
        """
        Computes the best F1 score and threshold value for a given set of scores and ground truth break points.

        Args:
            scores (np.array): The array of scores.
            gt_break_points (List[int]): The list of ground truth break points.

        Returns:
            Tuple[float, float]: A tuple containing the best F1 score and the corresponding threshold value.
        """

        threshold_values = scores[scores > 0]

        if len(threshold_values) == 0:
            return 0.0, 0.0
        
        threshold_values = np.unique(threshold_values)[::-1]

        best_f1 = 0
        max_fp = len(scores)
        best_index = 0

        for i_threshold, threshold in enumerate(threshold_values):

            # indices = np.where(scores >= threshold)[0]
            indices = np.where(scores > threshold)[0]
            false_positive = 0
            true_positive = 0 

            if len(indices) == 0:
                f1 = 0
            else:
                not_matched_positive = indices.copy()
                for p in gt_break_points:
                    if len(not_matched_positive) == 0:
                        break
                    if np.min(np.abs(not_matched_positive-p)) < self.peak_tolerance:
                        i = np.argmin(np.abs(not_matched_positive-p))
                        not_matched_positive = np.delete(not_matched_positive, i)

                false_positive = len(not_matched_positive)
                true_positive = len(indices) - false_positive

                if true_positive == 0:
                    f1 = 0
                else:
                    precision = true_positive / len(indices)
                    recall = true_positive / len(gt_break_points)

                    f1 = 2*precision * recall / (precision + recall)

            if f1 > best_f1:
                best_f1 = f1
                max_fp = 2*len(gt_break_points) / best_f1 - 2*len(gt_break_points)
                best_index = i_threshold
            else:
                if false_positive > max_fp:
                    break

        
        if best_index+1 < len(threshold_values):
            best_threshold = (threshold_values[best_index] + threshold_values[best_index+1])/2
        else:
            best_threshold = (threshold_values[best_index] + threshold_values[best_index-1])/2
        # best_threshold = threshold_values[best_index]

        return 1 - best_f1, best_threshold
    
    def _compute_f1(self, scores: np.array, gt_break_points: List[int], threshold: float) -> Tuple[float, float]:
        """
        Computes the F1 score for a given set of scores, ground truth break points and a fixed threhsold.

        Args:
            scores (np.array): The array of scores.
            gt_break_points (List[int]): The list of ground truth break points.

        Returns:
            Tuple[float, float]: A tuple containing the F1 score and the corresponding threshold value.
        """

        indices = np.where(scores >= threshold)[0]
        false_positive = 0
        true_positive = 0

        if len(indices) == 0:
            f1 = 0
        else:
            not_matched_positive = indices.copy()
            for p in gt_break_points:
                if len(not_matched_positive) == 0:
                    break
                if np.min(np.abs(not_matched_positive-p)) < self.peak_tolerance:
                    i = np.argmin(np.abs(not_matched_positive-p))
                    not_matched_positive = np.delete(not_matched_positive, i)

            false_positive = len(not_matched_positive)
            true_positive = len(indices) - false_positive

            if true_positive == 0:
                f1 = 0
            else:
                precision = true_positive / len(indices)
                recall = true_positive / len(gt_break_points)

                f1 = 2*precision * recall / (precision + recall)
              
        return 1 - f1, threshold

    def __call__(self, score: np.array, gt_break_points: List[int], threshold: float) -> float:
        """
        Calculates the F1 loss for a given set of scores and ground truth break points.

        Args:
            score (np.array): The array of scores.
            gt_break_points (List[int]): The list of ground truth break points.

        Returns:
            float: The F1 loss value.
        """
        f1, threshold = self._compute_f1(score, gt_break_points, threshold)
        return f1, threshold

    def get_optimal_threshold(self, score: np.array, gt_break_points: List[int]) -> float:
        """
        Calculates the optimal threshold value for a given set of scores and ground truth break points.

        Args:
            score (np.array): The array of scores.
            gt_break_points (List[int]): The list of ground truth break points.

        Returns:
            float: The optimal threshold value.
        """
        _, threshold = self._compute_best_f1(score, gt_break_points)
        self.threshold = threshold
        return threshold
    
    def get_minimum_value(self) -> float:
        """
        Returns the minimum possible value for the F1 loss.

        Returns:
            float: The minimum value for the F1 loss.
        """
        return 0

class AUCLoss(Loss):
    """
    AUCLoss is a class that calculates the rank-based surrogate loss for AUC using pairwise hing loss for a given set of scores and ground truth break points. 
    The AUC loss is defined as 1-AUC. This class has no get optimal threshold since no threshold is needed.
    """

    def __init__(self, peak_tolerance: int = 100):
        """
        Args:
            peak_tolerance (int): Tolerance window around each true break point.
        """
        self.peak_tolerance = peak_tolerance
    
    def __call__(self, scores: np.ndarray, gt_break_points: List[int]) -> float:
        N = len(scores)

        pos_mask = np.zeros(N, dtype=bool)
        for p in gt_break_points:
            start = max(0, p-self.peak_tolerance)
            end = min(N, p+self.peak_tolerance)
            pos_mask[start:end] = True
        neg_mask = ~pos_mask

        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]

        # if no positives or negatives, return minimum loss
        if len(pos_scores) == 00 or len(neg_scores) == 0:
            return 0.0
        
        diffs = pos_scores[:, None] - neg_scores[None, :]
        hinge_losses = np.maximum(0, 1 - diffs)

        return float(np.mean(hinge_losses))
    
    def get_optimal_threshold(self, score: np.array, gt_break_points: List[int]) -> float:
        """
        Calculate the optimal threshold value given the predicted score and ground truth break points.

        Args:
            score (np.array): The predicted score.
            gt_break_points (List[int]): The ground truth break points.

        Returns:
            float: The calculated optimal threshold value.
        """
        pass
    
    def get_minimum_value(self) -> float:
        """
        Returns the minimum possible value for the AUC loss.

        Returns:
            float: The minimum value for the AUC loss.
        """
        return 0.0
    
class ScoreModel(ABC):
    """
    Abstract base class for score models used in segmentation.
    """

    @property
    @abstractmethod
    def num_samples(self) -> int:
        """
        Get the number of samples used in the score model.

        Returns:
            int: The number of samples.
        """
        pass

    @property
    @abstractmethod
    def weights(self) -> List[float]:
        """
        Get the weights used in the score model.

        Returns:
            List[float]: The weights.
        """
        pass

    @abstractmethod
    def get_score(self) -> np.array:
        """
        Calculate the score using the score model.

        Returns:
            np.array: The calculated score.
        """
        pass

    @abstractmethod
    def get_weights_constraints(self) -> List[Tuple[float, float]]:
        """
        Get the constraints for the weights used in the score model.

        Returns:
            List[Tuple[float, float]]: The constraints for the weights.
        """
        pass

class WaveletDecompositionModel(ScoreModel):
    """
    A class representing a wavelet decomposition model for segmentation. 
    It performs wavelet decomposition on the input samples using the specified 
    level and wavelet type (db3). For each subband obtained from the decomposition, 
    it calculates a profile using the NormalDivergenceProfile, resamples it, 
    applies a power transform, and stores it. It then computes an average score 
    profile across all subbands and sets this as both the average and the score 
    profile of the model. Initial weights (that correspond to the exponents of
    the power transforms are set to 1 for each subband).

    Attributes:
        samples (np.array): The input samples for the model.
        level_wavelet (int): The level of wavelet decomposition.
        windows_size (int): The size of the windows used for profile calculations.
        average_profile (AverageScoreProfile): The average score profile of all subbands.
        score_profile (ProminenceProfile): The score profile of the model.
        _weights (List[float]): The weights assigned to each subband profile.
        _constraints (List[Tuple[float, float]]): The constraints for the weights.

    Methods:
        num_samples() -> int: Returns the number of samples.
        weights() -> List[float]: Returns the weights assigned to each subband profile.
        weights(w: List[float]): Sets the weights for each subband profile.
        get_score() -> np.array: Returns the score profile of the model.
        get_weights_constraints() -> List[Tuple[float, float]]: Returns the constraints for the weights.
    """

    def __init__(self, samples: np.array, level_wavelet: int = 7, windows_size: int = 100):
        """
        Initializes a new instance of the WaveletDecompositionModel class.

        Args:
            samples (np.array): The input samples for the model.
            level_wavelet (int, optional): The level of wavelet decomposition. Defaults to 7.
            windows_size (int, optional): The size of the windows used for profile calculations. Defaults to 100.
        """
        self.samples = samples
        self.level_wavelet = level_wavelet
        self.window_size = windows_size

        dec = pywt.wavedec(samples, wavelet='db2', level=level_wavelet, axis=0) 
        n_subband = len(dec)
        N = samples.shape[0]

        all_subband_profile = []
        weights = []
        constraints = []

        self.decomposition = dec
        self.div_profile = []
        self.resample_profile = []
        self.power_profile = []
        
        for _, subband in enumerate(dec):
            profile = NormalDivergenceProfile(subband, windows_size) 
            self.div_profile.append(profile)
            profile = ResampleProfile(profile, N)
            self.resample_profile.append(profile)
            w = 1 
            profile = PowerTransformProfile(profile, coeff=w)
            self.power_profile.append(profile)
            all_subband_profile.append(profile)
            weights.append(w)
            constraints.append((0.1, 10))

        profile = AverageScoreProfile(all_subband_profile, np.ones(n_subband))
        self.average_profile = profile
        profile = ProminenceProfile(profile)
        self.score_profile = profile
        self._weights = weights
        self._constraints = constraints
        self.all_subband_profile = all_subband_profile

    @property
    def num_samples(self) -> int:
        """
        Returns the number of samples.

        Returns:
            int: The number of samples.
        """
        return self.samples.shape[0]

    @property
    def weights(self) -> List[float]:
        """
        Returns the weights assigned to each subband profile.

        Returns:
            List[float]: The weights assigned to each subband profile.
        """
        return self._weights

    @weights.setter
    def weights(self, w: List[float]):
        """
        Sets the weights for each subband profile.

        Args:
            w (List[float]): The weights to be assigned to each subband profile.
        """
        self._weights = w
        for ww, profile in zip(self._weights, self.all_subband_profile):
            profile.coeff = ww

    def get_score(self) -> np.array:
        """
        Returns the score profile of the model.

        Returns:
            np.array: The score profile of the model.
        """
        return self.score_profile.get_profile()

    def get_allsubband_score(self) -> np.array:
        """
        Returns the set of scores of each subband
        
        Returns:
            np.array: The list of subband scores
        """
        all_subband_profile = []
        for subband in self.all_subband_profile:
            all_subband_profile.append(subband.get_profile())
        return all_subband_profile

    def get_weights_constraints(self) -> List[Tuple[float, float]]:
        """
        Returns the constraints for the weights.

        Returns:
            List[Tuple[float, float]]: The constraints for the weights.
        """
        return self._constraints


class Optimizer:
    """
    Abstract base class for optimizers. The optimizer is devoted to find the 
    optimal weights for a segmenter

    This class defines the interface for optimizers used in the segmentation process.
    Subclasses of `Optimizer` should implement the `update`, `set_segmenter`, and `set_loss` methods.

    Attributes:
        None

    Methods:
        update: Update the weights of the score model used by the segmenter and 
            return the updated weights.
        set_segmenter: Set the segmenter for the optimizer.
        set_loss: Set the loss function and minimum loss value for the optimizer.
    """

    @abstractmethod
    def update(self) -> List[float]:
        """
        Compute the new weights of the segmenter and return the computed weights

        Returns:
            List[float]: The computed weights.
        """
        pass

    @abstractmethod
    def set_segmenter(self, segmenter: "Segmenter"):
        """
        Sets the segmenter for the object.

        Args:
        - segmenter: An instance of the Segmenter class.

        """
        pass

    @abstractmethod
    def set_loss(self, loss: callable, minimum_loss_value: float = -np.inf):
        """
        Sets the loss function for the segmenter.

        Args:
            loss (callable): The loss function to be used for segmentation.
            minimum_loss_value (float): The minimum loss value allowed.

        """
        pass

class MangoOptimizer(Optimizer):
    """
    The MangoOptimizer class is used to optimize the weights of a segmenter using the Mango optimization algorithm.

    Attributes:
        loss (Loss): The loss function to be minimized.
        segmenter (Segmenter): The segmenter whose weights are being optimized.
        params_space (dict): A dictionary representing the parameter space for optimization.
        params_names (list): A list of parameter names.
        loss_to_adapt (callable): The loss function to be adapted for optimization.
        max_calls (int): The maximum number of function calls to be made during optimization.
        stored_points (list): A list of stored points for updating the optimizer.
        previuous_loss_value (float): The previous loss value.
        max_stored_points (int): The maximum number of stored points.

    Methods:
        update(): Updates the optimizer and returns the optimized weights.
        set_segmenter(segmenter: Segmenter): Sets the segmenter for optimization.
        set_loss(loss: callable, minimum_loss_value: float = -np.inf): Sets the loss function for optimization.
        objective(params_batch): Computes the objective values for a batch of parameter sets.
        early_stopping(results): Determines if early stopping criteria is met.

    """

    def __init__(self, max_calls=20, regularization=1):
        """
        Initializes a new instance of the MangoOptimizer class.

        Args:
            max_calls (int): The maximum number of function calls to be made during optimization. Default is 20.
        """
        self.loss: Loss = None
        self.regularization = regularization
        self.segmenter: Segmenter = None
        self.params_space = dict()
        self.params_names = []
        self.loss_to_adapt = None
        self.max_calls = max_calls
        self.n_updates = 0
        self.stored_points = []
        self.previuous_loss_value = None
        self.max_stored_points = max_calls // 4

    def update(self) -> np.array:
        """
        Updates the optimizer and returns the optimized weights.

        Returns:
            np.array: The optimized weights.

        """
        conf_dict = {
            'domain_size': 5000,
            'num_iteration': self.max_calls,
            'initial_custom': self.stored_points,
            'early_stopping': self.early_stopping,
        }
        
        tuner = Tuner(self.params_space, self.loss, conf_dict)
        res = tuner.minimize()

        self.previuous_loss_value = res['best_objective']

        _, stored_points = zip(*sorted(zip(res['objective_values'], res['params_tried']), key=lambda t: t[0]))
        stored_points = [p for p in stored_points]
        self.stored_points = stored_points[:min(self.max_stored_points, len(stored_points))]
        params = []
        for k in self.params_names:
            params.append(res['best_params'][k])
        return params[:-1], params[-1]

    def set_segmenter(self, segmenter: "Segmenter"):
        """
        Sets the segmenter for optimization.

        Args:
            segmenter (Segmenter): The segmenter to be optimized.

        """
        self.segmenter = segmenter

        weights = segmenter.weights
        weights_contraints = segmenter.get_weights_constraints()
        for i, c in enumerate(weights_contraints):
            name = f'w_{i}'
            self.params_names.append(name)
            self.params_space[name] = uniform(c[0], c[1]-c[0])
        
        params = dict()
        for name, w in zip(self.params_names, weights):
            params[name] = w
        
        threshold = segmenter.threshold
        name = f'threshold'
        self.params_names.append(name)
        self.params_space[name] = uniform(0, 5)
        params[name] = threshold

        self.stored_points = [params]
        

    def set_loss(self, loss: callable, minimum_loss_value: float = -np.inf):
        """
        Sets the loss function for optimization.

        Args:
            loss (callable): The loss function to be optimized.
            minimum_loss_value (float, optional): The minimum loss value. Defaults to -np.inf.

        """
        self.loss_to_adapt = loss
        self.loss = self.objective
        self.previuous_loss_value = minimum_loss_value
    
    def objective(self, params_batch):
        """
        Computes the objective values for a batch of parameter sets.

        Args:
            params_batch: A batch of parameter sets.

        Returns:
            list: The objective values for the parameter sets.

        """
        values = []
        for params in params_batch:
            params = [params[name] for name in self.params_names]
            value, _ = self.loss_to_adapt(params[:-1], params[-1]) + (self.regularization**self.n_updates)*np.linalg.norm(params[:-1], ord=2)
            # value, _ = self.loss_to_adapt(weights)
            values.append(value)
        return values
    
    def early_stopping(self, results):
        """
        Determines if early stopping criteria is met.

        Args:
            results: The optimization results.

        Returns:
            bool: True if early stopping criteria is met, False otherwise.

        """
        # results['best_objective'] < self.previuous_loss_value or
        return  results['best_objective'] == self.segmenter.loss.get_minimum_value()

class SupervisedDomain:
    """
    Represents a supervised domain for segmentation tasks, where specific intervals 
    in a data sequence are marked as supervised.

    Attributes:
        nsamples (int): Total number of samples in the domain.
        intervals (List[List[int]]): List of intervals (as [start, end)) representing supervised regions.

    Methods:
        __init__(nsamples: int): Initializes the supervised domain with the total sample count.
        __contains__(x: int) -> bool: Checks if a given sample index belongs to a supervised interval.
        __len__() -> int: Returns the total number of supervised samples.
        add_interval(interval: List[int]): Adds a new interval and resolves overlaps.
        _resolve_overlaps(): Merges overlapping or contiguous intervals.
        get_supervised_indices() -> np.array: Returns a boolean array indicating supervised samples.
        get_unsupervised_indices() -> np.array: Returns a boolean array for unsupervised samples.
    """
    
    def __init__(self, nsamples: int):
        self.nsamples = nsamples
        self.intervals = []  # Each interval is represented as [start, end)

    def __contains__(self, x: int) -> bool:
        """
        Checks if a sample index x is within any supervised interval.
        
        Args:
            x (int): The sample index to check.
        
        Returns:
            bool: True if x is supervised, False otherwise.
        """
        return any(start <= x < end for start, end in self.intervals)
    
    def __len__(self) -> int:
        """
        Returns the total number of supervised samples by summing the length of each interval.
        
        Returns:
            int: Number of supervised samples.
        """
        return sum(end - start for start, end in self.intervals)

    def add_interval(self, interval: list[int]):
        """
        Adds a new interval to the supervised domain and merges overlapping intervals.
        The interval is clamped to the bounds [0, nsamples).
        
        Args:
            interval (List[int]): An interval [start, end) to add.
        """
        start = max(0, interval[0])
        end = min(self.nsamples, interval[1])
        if start < end:  # Only add valid intervals
            self.intervals.append([start, end])
            self._resolve_overlaps()

    def _resolve_overlaps(self):
        """
        Resolves any overlapping intervals in the domain and fix new ground truth break points according to added intervals
        """
        self.intervals.sort(key=lambda x: x[0])
        new_intervals = []
        for interval in self.intervals:
            if not new_intervals:
                new_intervals.append(interval)
            else:
                last_interval = new_intervals[-1]
                if interval[0] <= last_interval[1]:
                    last_interval[1] = max(last_interval[1], interval[1])
                else:
                    new_intervals.append(interval)
        self.intervals = new_intervals

    def get_supervised_indices(self) -> np.array:
        """
        Returns a boolean array indicating which samples are supervised.

        Returns:
            np.array: A boolean array indicating which samples are supervised.
        """
        indices = np.zeros(self.nsamples, dtype=bool)
        for interval in self.intervals:
            try:
                indices[interval[0]:interval[1]] = True
            except:
                print(interval[0], interval[1])
        return indices
    
    def get_unsupervised_indices(self) -> np.array:
        """
        Returns a boolean array indicating which samples are unsupervised.

        Returns:
            np.array: A boolean array indicating which samples are unsupervised.
        """
        indices = self.get_supervised_indices()
        return ~indices

class Segmenter:
    """
    A class representing a segmenter for segmentation tasks. The segmenter is used to perform segmentation on a datastream using a score model. The segmenter can be compiled with a loss function and an optimizer to find the optimal weights for the score model.

    Attributes:
        score_model (ScoreModel): The score model used by the segmenter.
        gt_break_points (List[int]): The ground truth break points.
        supervised_domain (SupervisedDomain): The supervised domain for the segmenter.
        weights (List[float]): The weights assigned to the score model.
        threshold (float): The threshold value for segmentation.
        extension_window (int): The extension window for supervised intervals.
        loss (Loss): The loss function used by the segmenter.
        optimizer (Optimizer): The optimizer used by the segmenter.

    """

    def __init__(self, score_model: ScoreModel, extension_window: int = 100):
        """
        Initializes a new instance of the Segmenter class.

        Args:
            score_model (ScoreModel): The score model used by the segmenter.
            extension_window (int, optional): The extension window for supervised intervals. Defaults to 100.
        """
        
        self.score_model: ScoreModel = score_model
        self.gt_break_points = []
        self.supervised_domain = SupervisedDomain(score_model.num_samples)
        self.weights = score_model.weights

        # self.threshold = np.sort(score_model.get_score())[-1]
        # self.threshold = np.quantile(score_model.get_score(), 0.99)
        self.threshold = self.set_initial_threshold()
        self.extension_window = extension_window
        self.loss = None
        self.optimizer = None

    def compile(self, loss: Loss, optimizer: Optimizer):
        """
        Compiles the segmenter with a loss function and an optimizer. The loss function is used to calculate the loss value for the segmenter, while the optimizer is used to find the optimal weights for the score model.

        Args:
            loss (Loss): The loss function to be used for segmentation.
            optimizer (Optimizer): The optimizer to be used for optimization.
        """
        self.loss = loss
        optimizer.set_segmenter(self)
        optimizer.set_loss(self.loss_fun, loss.get_minimum_value())
        self.optimizer = optimizer

    def set_initial_threshold(self, q = 0.0):
        """
        Set the initial threshold using the curvature, completely vectorial.
        """

        scores = self.score_model.get_score()

        y = np.sort(scores[self.score_model.get_score()>0])[::-1]
        if len(y) < 5:
            return np.median(y) if len(y) > 0 else 0.0
        
        qval = np.percentile(y, 100*q)
        y = y[y >= qval]
        x = np.linspace(-1, 1, len(y))

        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())

        dy_dx = np.gradient(y_norm, x_norm)
        d2y_dx2 = np.gradient(dy_dx, x_norm)

        curvature = np.abs(d2y_dx2) / (1 + dy_dx**2)**1.5

        valid = curvature[2:-2]
        idx_offset = 2 + np.argmax(valid)

        threshold = y[idx_offset]
        return threshold
    
    @property
    def num_samples(self) -> int:
        """
        Returns the number of samples used in the segmenter.

        Returns:
            int: The number of samples.
        """
        return self.score_model.num_samples

    def loss_fun(self, weights: List[float], threshold: float):
        """
        Calculates the loss value for the segmenter using the given weights.

        Args:
            weights (List[float]): The weights to be used for segmentation.
        Returns:
            float: The calculated loss value.
        """
        score = self.get_supervised_score(weights)
        gt_break_points = self.gt_break_points
        loss_value = self.loss(score, gt_break_points, threshold)
        return loss_value

    def get_optimal_threshold(self, score: np.array = None) -> float:
        """
        Calculates the optimal threshold value for the segmenter.

        Returns:
            float: The optimal threshold value.
        """
        if score is None:
            score = self.get_supervised_score()
        gt_break_points = self.gt_break_points
        return self.loss.get_optimal_threshold(score, gt_break_points)
        
    def get_score(self, weights: List[float] = None) -> np.array:
        """
        Calculates the score for the segmenter using the given weights.

        Args:
            weights (List[float], optional): The weights to be used for segmentation. Defaults to None. If None is provided, the weights stored in the segmenter are used.
        
        Returns:
            np.array: The calculated score.
        """
        if weights is None:
            weights = self.weights
        self.score_model.weights = weights
        return self.score_model.get_score()
    
    @property
    def samples(self) -> np.array:
        """
        Returns the samples used by the segmenter.

        Returns:
            np.array: The samples.
        """
        return self.score_model.samples
        
    def get_supervised_score(self, weights: List[float] = None) -> np.array:
        """
        Calculates the score for the segmenter using the given weights, considering only the supervised regions.

        Args:
            weights (List[float], optional): The weights to be used for segmentation. Defaults to None. If None is provided, the weights stored in the segmenter are used.

        Returns:
            np.array: The calculated score.
        """
        score = self.get_score(weights) * self.supervised_domain.get_supervised_indices()
        return score
    
    def get_unsupervised_score(self, weights: List[float] = None) -> np.array:
        """
        Calculates the score for the segmenter using the given weights, considering only the unsupervised regions.

        Args:
            weights (List[float], optional): The weights to be used for segmentation. Defaults to None. If None is provided, the weights stored in the segmenter are used.

        Returns:
            np.array: The calculated score.
        """
        score = self.get_score(weights) * self.supervised_domain.get_unsupervised_indices()
        return score

    def update(self):
        """
        Updates the segmenter using the optimizer. The weights of the score model are updated using the optimizer, and the optimal threshold value is calculated. Weights are updated if and only if a new gt_break_point
        has been added to the ground_truth, if not, only the threshold optimization is runned.
        """
        self.weights, self.threshold = self.optimizer.update()
        # self.threshold = self.get_optimal_threshold()

    def get_weights_constraints(self) -> List[Tuple[float, float]]:
        """
        Returns the constraints for the weights used in the segmenter.

        Returns:
            List[Tuple[float, float]]: The constraints for the weights.
        """
        return self.score_model.get_weights_constraints()

    def get_break_points(self) -> List[int]:
        """
        Returns the break points detected by the segmenter.

        Returns:
            List[int]: The detected break points.
        """
        bkps = np.where(self.get_unsupervised_score() >= self.threshold)[0]
        #combine with gt break points
        bkps = np.concatenate((bkps, self.gt_break_points))
        #remove duplicates
        bkps = np.sort(np.unique(bkps))
        return bkps

    def add_break_point_to_gt(self, break_point: int, supervised_interval: List[int]):
        """
        Adds a break point to the ground truth break points and the supervised domain.

        Args:
            break_point (int): The break point to add.
            supervised_interval (List[int]): The supervised interval for the break point.
        """
        if not (supervised_interval[0] <= break_point <= supervised_interval[1]):
            return ValueError("The break point must be inside the supervised interval")
        break_points = self.gt_break_points
        break_points.append(break_point)
        self.gt_break_points = list(sorted([b for b in break_points]))
        self.supervised_domain.add_interval(supervised_interval)

    def add_supervised_interval(self, supervised_interval: List[int]): 
        """
        Adds a supervised interval to the segmenter.

        Args:
            supervised_interval (List[int]): The supervised interval to add.
        """
        self.supervised_domain.add_interval(supervised_interval)

    def get_least_certain(self, score: np.array = None, threshold: float = None):
        """
        Get least certain sample to query the user.

        Args:
        """
        if score is None:
            score = self.get_unsupervised_score()
        if threshold is None:
            threshold = self.threshold

        if threshold > score.max():
            threshold = score.max() - 1e-6
        if threshold < score.min():
            threshold = score.min() + 1e-6

        pos_idx = np.nonzero(score>0)[0]
        if pos_idx.size == 0: 
            if len(np.nonzero(self.supervised_domain.get_supervised_indices())[0]) == 0:
                # return np.random.choice(np.arange(self.num_samples), size=2, replace=False)
                print('$$$$$$$$$$  Empty Unsupervised Domain  $$$$$$$$$$$$')
                return 0
            return np.random.choice(np.nonzero(self.supervised_domain.get_unsupervised_indices())[0], size=2, replace=False)
        
        pos_scores = score[pos_idx]

        below_mask = pos_scores <= threshold
        if np.any(below_mask):
            below_scores = pos_scores[below_mask]
            below_idx_local = np.argmax(below_scores)
            below_idx = pos_idx[below_mask][below_idx_local]
        else:
            below_idx = np.random.choice(self.supervised_domain.get_unsupervised_indices(), size=1, replace=False)
        
        above_mask = pos_scores >= threshold
        if np.any(above_mask):
            above_scores = pos_scores[above_mask]
            above_idx_local = np.argmin(above_scores)
            above_idx = pos_idx[above_mask][above_idx_local]
        else:
            above_idx = np.random.choice(self.supervised_domain.get_unsupervised_indices(), size=1, replace=False)

        return below_idx, above_idx
    
   

    def get_metrics(self, break_points: List[int], gt_break_points: List[int]) -> Tuple[float, float, float]:
        """
        Calculates the precision, recall, and F1 score for the segmenter.        
        Args:
            scores (np.array): The scores of the segmenter.
            gt_break_points (List[int]): The ground truth break points.
        Returns:
            Tuple[float, float, float]: A tuple containing the precision, recall, and F1 score.
        """
        indices = break_points
        true_positive = 0
        false_positive = 0

        if len(indices) == 0:
            precision = 0
            recall = 0
        else:
            not_matched_positive = indices.copy()
            for p in gt_break_points:
                if len(not_matched_positive) == 0:
                    break
                if np.min(np.abs(not_matched_positive-p)) < self.loss.peak_tolerance:
                    i = np.argmin(np.abs(not_matched_positive-p))
                    not_matched_positive = np.delete(not_matched_positive, i)
                
            false_positive = len(not_matched_positive)
            true_positive = len(indices) - false_positive

            if true_positive == 0:
                precision = 0
                recall = 0
            else:
                precision = true_positive / len(indices)
                recall = true_positive / len(gt_break_points)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2*precision * recall / (precision + recall)
        
        return precision, recall, f1

    def saver_config(self, path_prefix):

        with open(f'{path_prefix}_segmenter.pkl', 'wb') as f:
            pickle.dump(self, f)

        config = {
            'level_wavelet': self.score_model.level_wavelet,
            'window_size': self.score_model.window_size,
            'threshold': self.threshold,
            'weights': self.weights
        }
        with open(f'{path_prefix}_config.json', 'w') as f:
            json.dump(config, f, indent=4)
    
    @classmethod
    def load_from_config(cls, path, samples):
        with open(path, 'r') as f:
            config = json.load(f)
        
        score_model = WaveletDecompositionModel(samples, windows_size=config['window_size'], level_wavelet=config['level_wavelet'])
        score_model.weights = config['weights']
        segmenter = Segmenter(score_model=score_model)
        segmenter.threshold = config['threshold']

        return score_model, segmenter