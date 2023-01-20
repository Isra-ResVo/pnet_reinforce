import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import logging

from evaluation.data_point import Point


def extraElements(batchShape, device, norm, oneElement=False) -> torch.Tensor:

    """create extra elements for be concatened in the batch tensor
    only works when batch is normlized, because it takes values for represent
    this special elements outside of [0,1] range, where is the normal elements domain
    """
    a, b, _ = batchShape
    elementsToGenerate = 2  # default value

    if oneElement:
        elementsToGenerate = 1

    extraEle = torch.ones(
        (a, b, elementsToGenerate), dtype=torch.float32, device=device
    )

    extraEle[:, :, 0] = 1.5

    if not oneElement:
        extraEle[:, :, 1] = 2

    return extraEle


def normalization(batch: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
    Normalize function -> (0,1)
    using the next formula by element and caracteristic
        element - minimum
        -----------------
        maximum - minimum

    this works with tensor arrange of the next way
        [batch index, caracteristic, elements]
    """

    batchNorm = torch.zeros_like(batch)  # default device of input
    for i, element in enumerate(batch):
        for j, parameter in enumerate(element):
            mini = min(parameter)
            maxi = max(parameter)
            batchNorm[i][j] = (parameter - mini) / (maxi - mini)

    return batchNorm


# labels for plotting
def labelsFunc(point: Point, mode):
    """

    This funtion create the the xticks labels this take place intead
    of the values in the x-axis. This will be the combination of clouds
    and level of redundancy example: [(2,1), (2,2),... ]

    """
    n = point.n_inferred
    siz = point.elements_length

    labels = []
    if mode == "n":
        for i in range(2, n + 1):
            labels.append((i, n.item()))

    elif mode == "k_n":
        cloud_qnts = torch.arange(start=2, end=siz + 1, device="cpu")
        ks = [
            torch.arange(start=2, end=cloud_qnt + 1, device="cpu")
            for cloud_qnt in cloud_qnts
        ]

        # tuple values for x's tick for matplotlib
        labels = []
        for c, b in zip(cloud_qnts, ks):
            for z in b:
                labels.append((z.item(), c.item()))

    elif mode == "k":
        k = point.k_inferred
        labels = [(k.item(), i) for i in range(k, siz + 1)]

    else:
        raise RuntimeError("arg:mode must to be 'n', k or 'k_n' ")

    return labels


def plotting(
    data_and_names,
    pointsToGraph=None,
    mode: str = None,
    path=None,
    logScale=False,
    labeloptions: tuple = (False, "upper right"),
    annotatewo=True,
    point: Point = None,
) -> None:

    # points2graph tuple (value, name2display)
    # data_and_name

    legendingraph, location = labeloptions
    anotaciones = True

    # Validation
    log = logging.getLogger(__name__)

    # Raise error
    if not (mode is None or type(mode) is str):
        raise TypeError("Mode is getting a invalid type")

    # Validate that all data have the same length
    to_compare = data_and_names[0][0].shape[0]  # first data element
    for i, (data, name) in enumerate(data_and_names):
        if data.shape[0] != to_compare:
            print(sys.exc_info()[0])
            raise ValueError("Tensor doesn't have the same dimention")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    linecolor = None
    for element, name in data_and_names:
        if isinstance(element, torch.Tensor):
            element = element.cpu()
        if name == "Redundancia":
            linecolor = "#ff7f0e"
        elif name == "Probabilidad de perdida":
            linecolor = "#1f77b4"
        elif name == "WO":
            linecolor = "#2ca02c"
        else:
            linecolor = None

        ax.plot(element, marker="o", label=name, color=linecolor)

        # Annotations to the points (putting the values in the graph for each data point)
        if anotaciones and not (name == "WO" and not annotatewo):
            for i, val in enumerate(element):
                if isinstance(val, torch.Tensor):
                    val = val.item()
                if logScale:
                    txt = "{:.2e}".format(val)
                else:
                    txt = "{:.2f}".format(val)

                if "WO" == name:
                    if name == "WO":
                        rotate = 90
                    else:
                        rotate = 45
                else:
                    rotate = 0
                ax.annotate(
                    txt,
                    xy=(i, val),
                    xytext=(0, 1),  # One point of vertical offset
                    textcoords="offset points",
                    rotation=rotate,
                    ha="center",
                    va="bottom",
                )

    # necesary data for calculate position and xticks
    n = point.n_inferred.float()
    k = point.k_inferred.float()
    siz = point.elements_length

    if mode == "k_n":
        position = (((n - 2) * (n - 1)) / 2) + k - 2
        elements = ((siz - 1) * (siz)) / 2
        val_ticks = torch.arange(0, elements).cpu()

    elif mode == "n":
        position = k - 2
        val_ticks = torch.arange(0, n - 1).cpu()
    elif mode == "k":
        position = n - k
        val_ticks = torch.arange(0, siz - k + 1)

    annotations = True
    if pointsToGraph is not None and annotations:
        for (val, name) in pointsToGraph:
            txt = "Model"
            plt.annotate(txt, xy=(position, val), arrowprops=dict(facecolor="black"))

    labels = labelsFunc(point, mode)
    if labels is not None:
        plt.xticks(val_ticks, labels, rotation="vertical", fontsize=8)

    title = path.split("/")[2].split(".")[0]
    ax.set(xlabel="Valores validos de la tupla (k,n)", ylabel="Valores", title=title)
    ax.grid()

    if logScale:
        plt.yscale("log")

    plt.rcParams.update({"font.size": 8})
    plt.subplots_adjust(bottom=0.25, top=0.95, right=0.98, left=0.1)
    # plt.axis('off')
    if legendingraph:
        plt.legend(loc=location)
    else:
        plt.legend(bbox_to_anchor=(0, -0.26), loc="center left", borderaxespad=0.0)

    # save fig
    plt.savefig(path)
    plt.show()
    plt.close()
