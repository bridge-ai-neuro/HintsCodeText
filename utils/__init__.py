from utils.plot_utils import *
from numpy import std, mean, sqrt

HINT_MAPPING = {
    "none": "No Hint",
    "testcases": "Test Cases",
    "conceptualhint": "Conceptual",
    "detailedfix": "Detailed Fix",
}
TASK_NUM_TO_NAME = {
    "1": "Sum Positives",
    "2": "Count Pos Neg",
    "3": "Average Rainfall",
    "4": "Palindrome",
    "5": "Fibonacci",
}

HINT_ORDER = list(HINT_MAPPING.values())
TASK_ORDER = list(TASK_NUM_TO_NAME.values())
STIMULUS_ORDER = ["Python", "Text"]
PART_ORDER = ["Confused", "Clear"]


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(
        ((nx - 1) * std(x, ddof=1) ** 2 + (ny - 1) * std(y, ddof=1) ** 2) / dof
    )
