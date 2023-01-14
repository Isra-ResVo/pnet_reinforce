import torch
import numpy as np
from evaluation.data_point import Point
from reward.reward_representation import DataSolution


class Statistics:
    def __init__(self, reward_grouped: DataSolution):
        self.reward_grouped = reward_grouped
        self.val_statistic = {
            "pr_ln": [],
            "prn_ln": [],
            "redundancy": [],
            "rn": [],
            "wo": [],
            "won": [],
        }

    def add_values(self, point_object: Point, index: int):
        self.val_statistic["pr_ln"].append(point_object.probability_of_error)
        self.val_statistic["wo"].append(point_object.weighted_objective)
        self.val_statistic["won"].append(point_object.weighted_objective) # revisar
        self.val_statistic["redundancy"].append(
            self.reward_grouped.normalized_redundancy[index]
        )
        self.val_statistic["prn_ln"].append(
            point_object.probability_of_error_normalized
        )
        self.val_statistic["rn"].append(self.reward_grouped.normalized_redundancy[index])

    def print_results(self):

        print("Inferred values of n and k  for every element in the batch")
        for i, (val_k, val_n) in enumerate(
            zip(self.reward_grouped.k_inferred, self.reward_grouped.n_inferred)
        ):
            print(
                "Element n*.{}, inferred values ({},{})".format(
                    i + 1, val_k, val_n
                )
            )
        
        for key in self.val_statistic:
            self.val_statistic[key] = torch.stack(self.val_statistic[key]).numpy()

            print("\nStatisctic {}".format(key))
            if self.val_statistic[key].shape[0] <= 1:
                print(
                    "Single element: Cannot make any operation over it."
                )
            else:
                print("Media, Variance, Standard deviation")
                print(
                    "{:.4f}, {:.6f}, {:.4f}".format(
                        np.mean(self.val_statistic[key]),
                        np.var(self.val_statistic[key], ddof=1),
                        np.std(self.val_statistic[key], ddof=1),
                    )
                )
                print("Data used to calculate the values:\n", self.val_statistic[key])