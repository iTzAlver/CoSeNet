# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from implementations import Prepare, Predict, Fit, deep_clustering, modularity_maximization, \
    community_detection, herarchical_clustering, correlation_clustering, proximity_based_merging_means, segcorr


def deep_clustering_mlp(matrix, labels=None):
    return deep_clustering(matrix, labels=labels, type_enc='mlp')


def deep_clustering_cnn(matrix, labels=None):
    return deep_clustering(matrix, labels=labels, type_enc='cnn')


MODEL_FUNCTIONS = [
    modularity_maximization,
    community_detection,
    herarchical_clustering,
    correlation_clustering,
    proximity_based_merging_means,
    segcorr,
    deep_clustering_mlp,
    deep_clustering_cnn,
    # deep_clustering_unique,
]


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def main():
    prep = Prepare()
    prep_synth = Prepare(synth=True)
    initial_parameters = {
        'modularity_maximization': [(0.25,), (0, 1, float)],
        'community_detection': [(0.25,), (0, 1, float)],
        'herarchical_clustering': [(0.25, 10), (0, 1, float), (3, 20, int)],
        'correlation_clustering': [(0.25, 10), (0, 1, float), (3, 20, int)],
        'proximity_based_merging_means': [(0.25, 1, 0.3), (0, 1, float), (0, 4, int), (0, 1, float)],
        'deep_clustering_mlp': [()],
        'deep_clustering_cnn': [()],
        'deep_clustering_unique': [(20,), (10, 50, int)],
        'segcorr': [(0.5,), (0, 1, float)],
    }

    for model_function in MODEL_FUNCTIONS:
        # Fitting the model.
        fitter = Fit(model_function, prep.matrix[0], prep.labels[0],
                     parameters=initial_parameters[model_function.__name__])
        best_parameters = fitter.best_parameters
        # Testing the model.
        Predict(model_function, prep_synth.matrix[1:], prep_synth.labels[1:],
                parameters=best_parameters, synth=True)
        Predict(model_function, prep.matrix, prep.labels, parameters=best_parameters)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
