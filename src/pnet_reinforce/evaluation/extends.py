from extras import plotting
from reward.reward_representation import DataSolution
from generator.data_interface.data import DataRepresentation
from evaluation.statisctics import Statistics

from evaluation.data_point import Point
from evaluation.point_references.probability_error import ProbalityErrorReferences
from evaluation.point_references.weighted import WeightedObjectiveReferences
from evaluation.point_references.redundancy import RedundancyReferences


def extend_information(
    data_object: DataRepresentation,
    reward_grouped: DataSolution,
    config,
    path: str,
    plot: bool,
    printOpt: float,
):
    statistics = Statistics(reward_grouped=reward_grouped)
    batch = data_object.batch

    for index, _ in enumerate(batch):

        point_object = Point(
            reward_grouped=reward_grouped, data_object=data_object, index=index
        )

        tuples_data_and_names = []
        text = {}

        # Add probability of error references data to the point object.
        ProbalityErrorReferences(point_object=point_object).add(
            data_object=data_object, config=config, idxEle=index
        )

        # Add redundancy data references to the point object.
        RedundancyReferences(reward_config=reward_grouped.reward_config).add(
            point=point_object, config=config, data_object=data_object, index=index
        )

        # Add weighted objective data to the point object.
        WeightedObjectiveReferences(point_object=point_object, config=config).add()

        if "prError" in config.whatToGraph and plot:
            # cambio a peticion del Dr.
            local_name = ["Probabilidad de perdida"]

            tuple_pr_error_2graph = (
                point_object.all_element_batch_error_probabilities,
                point_object.all_element_batch_error_probabilities_no_log,
            )

            if config.mode == "n":
                labeloptions = (True, "upper left")
            else:
                labeloptions = (True, "lower left")

            pathLocal = path + " Probabilidad de perdida.png"

            if config.log:
                points2graph = [(point_object.probability_of_error, "pr_ln")]
                pr_error_2graph = tuple_pr_error_2graph[0]  # ln values

            else:
                points2graph = [(point_object.probability_of_error_no_log, "pr")]
                pr_error_2graph = tuple_pr_error_2graph[1]

            data_and_name_2graph = [(pr_error_2graph, local_name)]

            plotting(
                data_and_names=data_and_name_2graph,
                pointsToGraph=points2graph,
                point=point_object,
                mode=config.mode,
                path=pathLocal,
                logScale=True,
                labeloptions=labeloptions,
            )

            tuples_data_and_names.append(
                (
                    point_object.all_element_batch_error_probabilities_normalized,
                    local_name,
                )
            )

        if "redundancy" in config.whatToGraph and plot:

            localNames = ["Redundancia"]

            if config.mode == "n":
                labeloptions = (True, "upper right")
            else:
                labeloptions = (True, "upper left")

            pathLocal = path + " Redundancia.png"
            points2graph = [(point_object.redundancy, "redundancy")]
            data_and_name = [
                (point_object.redundancy_all_values_of_element_batch, localNames)
            ]

            plotting(
                data_and_name,
                points2graph,
                point=point_object,
                mode=config.mode,
                path=pathLocal,
                labeloptions=labeloptions,
            )

            tuples_data_and_names.append(
                (
                    point_object.redundancy_all_values_of_element_batch_normalized,
                    "Redundancia",
                )
            )

        if plot:
            woValues2Graph = point_object.all_element_batch_ponderate_objective
            tuples_data_and_names.append((woValues2Graph, "WO"))
            pathComparation = path + " Comparacion de objetivos.png"
            annotatewo = False if 1 in config.wo else True

            if config.monoObjetive is not None:
                if config.monoObjetive == "prError":
                    points2graph = [
                        (point_object.probability_of_error_normalized, "prn_nl")
                    ]
                elif config.monoObjetive == "redundancy":
                    points2graph = [(point_object.normalized_redundancy, "Rn")]
                else:
                    print(config.monoObjetive)
                    raise NotImplementedError(
                        "the value in monoObjetive is not valid only is implemented prError and redundancy"
                    )
            else:
                points2graph = [(point_object.weighted_objective, "WO")]

            plotting(
                tuples_data_and_names,
                points2graph,
                point=point_object,
                mode=config.mode,
                path=pathComparation,
                annotatewo=annotatewo,
            )

        if config.statistic:
            statistics.add_values(point_object=point_object, index=index)

    if config.statistic:
        statistics.print_results()
