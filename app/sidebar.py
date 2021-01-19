"""
Author: Jason Eisele
Date: December 21, 2020
Scope: Shows the sidebar for the streamlit app and manages all user inputs.
Email: jeisele@shipt.com
"""

import streamlit as st
import model_config

# Define possible models in a dict.
# Format of the dict:
# option 1: framework -> model -> code
# option 2 – if model has multiple variants: framework -> model -> model variant -> code
# option 3: framework -> problem -> model -> code

# MODELS = {
#     "PyTorch": {
#         "AlexNet": "alexnet",  # single model variant
#         "ResNet": {  # multiple model variants
#             "ResNet 18": "resnet18",
#             "ResNet 34": "resnet34",
#             "ResNet 50": "resnet50",
#             "ResNet 101": "resnet101",
#             "ResNet 152": "resnet152",
#         },
#         "DenseNet": "densenet",
#         "VGG": {
#             "VGG11": "vgg11",
#             "VGG11 with batch normalization": "vgg11_bn",
#             "VGG13": "vgg13",
#             "VGG13 with batch normalization": "vgg13_bn",
#             "VGG16": "vgg16",
#             "VGG16 with batch normalization": "vgg16_bn",
#             "VGG19": "vgg19",
#             "VGG19 with batch normalization": "vgg19_bn",
#         },
#     },
#     "scikit-learn": {
#         "Support vectors": "sklearn.svm.SVC",
#         "Random forest": "sklearn.ensemble.RandomForestClassifier",
#         "Decision tree": "sklearn.tree.DecisionTreeClassifier",
#         "Perceptron": "sklearn.linear_model.Perceptron",
#         "K-nearest neighbors": "sklearn.neighbors.KNeighborsClassifier",
#     },
# }
MODELS = {
    "scikit-learn": {
        "Supervised Learning": {
            "Linear Models": {
                "Ordinary Least Squares": "sklearn.linear_model.LinearRegression",
                "Ridge Regression": "sklearn.linear_model.Ridge",
                "Lasso Regression": "sklearn.linear_model.Lasso",
                "Multi-Task Lasso": "sklearn.linear_model.MultiTaskLasso",
                "Elastic Net": "sklearn.linear_model.ElasticNet",
                "Multi-Task Elastic Net": "sklearn.linear_model.MultiTaskElasticNet",
                "Least Angle Regression (LARS)": "sklearn.linear_model.Lars",
                "LARS Lasso": "sklearn.linear_model.LassoLars",
                "Orthogonal Matching Pursuit": "sklearn.linear_model.OrthogonalMatchingPursuit",
                "Bayesian Ridge Regression": "sklearn.linear_model.BayesianRidge",
                "Automatic Relevance Determination (ARD)": "sklearn.linear_model.ARDRegression",
                "Logistic Regression": "sklearn.linear_model.LogisticRegression",
                "Tweedie Regression": "sklearn.linear_model.TweedieRegressor",
                "Stochastic Gradient Descent Regressor": "sklearn.linear_model.SGDRegressor",
                "Perceptron": "sklearn.linear_model.Perceptron",
                "Passive Aggressive Regression": "sklearn.linear_model.PassiveAggressiveRegressor",
                "Passive Aggressive Classification": "sklearn.linear_model.PassiveAggressiveClassifier",
                "RANSAC (RANdom SAmple Consensus)": "sklearn.linear_model.RANSACRegressor",
                "Theil Sen Regression": "sklearn.linear_model.TheilSenRegressor",
                "Huber Regression": "sklearn.linear_model.HuberRegressor"
            },
            "Linear and Quadratic Discriminant Analysis": {
                "Linear Discriminent Analysis": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
                "Quadratic Discriminant Analysis": "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis"
            },
            "Kernel Ridge Regression": {
                "Kernel Ridge Regression": "sklearn.kernel_ridge.KernelRidge"
            },
            "Support Vector Machines": {
                "Support Vector Classification": "sklearn.svm.SVC",
                "Nu-Support Vector Classification": "sklearn.svm.NuSVC",
                "Linear Support Classification": "sklearn.svm.LinearSVC",
                "Support Vector Regression": "sklearn.svm.SVR",
                "Nu-Support Vector Regression": "sklearn.svm.NuSVR",
                "Linear Support Regression": "sklearn.svm.LinearSVR"
            },
            "Stochastic Gradient Descent": {
                "Stochastic Gradient Descent Classification": "sklearn.linear_model.SGDClassifier",
                "Stochastic Gradient Descent Regression": "sklearn.linear_model.SGDRegressor"
            },
            "Nearest Neighbors": {
                "Nearest Neighbors": "sklearn.neighbors.NearestNeighbors",
                "KDTree": "sklearn.neighbors.KDTree",
                "Ball Tree": "sklearn.neighbors.BallTree",
                "K-Nearest Neighbors Classification": "sklearn.neighbors.KNeighborsClassifier",
                "Radius Neighbors Classification": "sklearn.neighbors.RadiusNeighborsClassifier",
                "K Neighbors Regression": "sklearn.neighbors.KNeighborsRegressor",
                "Radius Neighbors Regression": "sklearn.neighbors.RadiusNeighborsRegressor",
                "Nearest Centroid Classification": "sklearn.neighbors.NearestCentroid",
            },
            "Gaussian Processes": {
                "Gaussian Process Regression (GPR)": "sklearn.gaussian_process.GaussianProcessRegressor",
                "Gaussian Process Classification (GPC)": "sklearn.gaussian_process.GaussianProcessClassifier",
            },
            "Cross decomposition": {
                "Partial Least Squares Regression": "sklearn.cross_decomposition.PLSRegression",
                "Partial Least Squares Canonical": "sklearn.cross_decomposition.PLSCanonical"
            },
            "Naive Bayes": {
                "Gaussian Naive Bayes Classification": "sklearn.naive_bayes.GaussianNB",
                "Multi-Nomial Naive Bayes Classification": "sklearn.naive_bayes.MultinomialNB",
                "Complement Naive Bayes Classification": "sklearn.naive_bayes.ComplementNB",
                "Bernoulli Naive Bayes Classification": "sklearn.naive_bayes.BernoulliNB",
                "Categorical Naive Bayes Classification": "sklearn.naive_bayes.CategoricalNB",
            },
            "Decision Trees": {
                "Decision Tree Classification": "sklearn.tree.DecisionTreeClassifier",
                "Decision Tree Regression": "sklearn.tree.DecisionTreeRegressor",
            },
            "Ensemble Methods": {
                "Bagging Classification": "sklearn.ensemble.BaggingClassifier",
                "Bagging Regression": "sklearn.ensemble.BaggingRegressor",
                "Random Forest Classification": "sklearn.ensemble.RandomForestClassifier",
                "Random Forest Regression": "sklearn.ensemble.RandomForestRegressor",
                "Extra Trees Classification": "sklearn.ensemble.ExtraTreesClassifier",
                "Extra Trees Regression": "sklearn.ensemble.ExtraTreesRegressor",
                "AdaBoost Classification": "sklearn.ensemble.AdaBoostClassifier",
                "AdaBoost Regression": "sklearn.ensemble.AdaBoostRegressor",
                "Histogram-based Gradient Boosting Classification Tree": "sklearn.ensemble.HistGradientBoostingClassifier",
                "Histogram-based Gradient Boosting Regression Tree": "sklearn.ensemble.HistGradientBoostingRegressor",
                "Gradient Boosting Classification": "sklearn.ensemble.GradientBoostingClassifier",
                "Gradient Boosting Regression": "sklearn.ensemble.GradientBoostingRegressor",
                "Voting Classifier": "sklearn.ensemble.VotingClassifier",
                "Voting Regression": "sklearn.ensemble.VotingRegressor",
                "Stacking Classifier": "sklearn.ensemble.StackingClassifier",
                "Stacking Regression": "sklearn.ensemble.StackingRegressor",
            },
        },
        "Unsupervised Learning": {
            "Gaussian Mixture Models": {
                "Gaussian Mixture": "sklearn.mixture.GaussianMixture",
                "Variational Bayesian Gaussian Mixture": "sklearn.mixture.BayesianGaussianMixture"
            },
            "Manifold Learning": {
                "Isomap": "sklearn.manifold.Isomap",

            },
            "Clustering": {
                "K-Means Clustering": "sklearn.cluster.KMeans",
                "Mini-Batch K-Means Clustering": "sklearn.cluster.MiniBatchKMeans",
                "Affinity Propagation": "sklearn.cluster.AffinityPropagation",
                "Mean Shift": "sklearn.cluster.MeanShift",
                "Spectral Clustering": "sklearn.cluster.SpectralClustering",
                "Agglomerative Clustering": "sklearn.cluster.AgglomerativeClustering",
                "Density-Based Spatial Clustering of Applications with Noise (DBSCAN)": "sklearn.cluster.DBSCAN",
                "Ordering Points To Identify the Clustering Structure (OPTICS)": "sklearn.cluster.OPTICS",
                "Birch": "sklearn.cluster.Birch",
            },
            "Biclustering": {
                "Spectral Bi-Clustering": "sklearn.cluster.SpectralBiclustering",
                "Spectral Co-Clustering": "sklearn.cluster.SpectralCoclustering"
            },
            "Decomposing Signals in Components (Matrix Factorization Problems)": {
                "Principal Component Analysis (PCA)": "sklearn.decomposition.PCA",
                "Kernel PCA": "sklearn.decomposition.KernelPCA",
                "Sparse PCA": "sklearn.decomposition.SparsePCA",
                "Mini-batch Sparse PCA": "sklearn.decomposition.MiniBatchSparsePCA",
                "Truncated Singular Value Decomposition": "sklearn.decomposition.TruncatedSVD",
                "Sparse Coding": "sklearn.decomposition.SparseCoder",
                "Dictionary Learning": "sklearn.decomposition.DictionaryLearning",
                "Mini-batch Dictionary Learning": "sklearn.decomposition.MiniBatchDictionaryLearning",
                "Factor Analysis": "sklearn.decomposition.FactorAnalysis",
                "Independent Component Analysis": "sklearn.decomposition.FastICA",
                "Non-negative matrix factorization": "sklearn.decomposition.NMF",
                "Latent Dirichlet Allocation (LDA)": "sklearn.decomposition.LatentDirichletAllocation",
            },
            "Covariance Estimation": {
                "Empirical Covariance": "sklearn.covariance.EmpiricalCovariance",
                "Shrunk Covariance": "sklearn.covariance.ShrunkCovariance",
                "Ledoit Wolf Covariance": "sklearn.covariance.LedoitWolf",
                "Graphical Lasso": "sklearn.covariance.GraphicalLasso",
                "Minimum Covariance Determinant (MCD)": "sklearn.covariance.MinCovDet"
            },
            "Novelty and Outlier Detection": {
                "Unsupervised Outlier Detection": "sklearn.svm.OneClassSVM",
                "Isolation Forest": "sklearn.ensemble.IsolationForest",
                "Unsupervised Outlier Detection using Local Outlier Factor (LOF)": "sklearn.neighbors.LocalOutlierFactor"
            },
            "Density Estimation": {
                "Kernel Density Estimation": "sklearn.neighbors.KernelDensity",
            },
            "Neural Network Models": {
                "Bernoulli Restricted Boltzmann Machine": "sklearn.neural_network.BernoulliRBM"
            }
        },
    },
    "PyTorch": {
        "AlexNet": "alexnet",  # single model variant
        "ResNet": {  # multiple model variants
            "ResNet 18": "resnet18",
            "ResNet 34": "resnet34",
            "ResNet 50": "resnet50",
            "ResNet 101": "resnet101",
            "ResNet 152": "resnet152",
        },
        "DenseNet": "densenet",
        "VGG": {
            "VGG11": "vgg11",
            "VGG11 with batch normalization": "vgg11_bn",
            "VGG13": "vgg13",
            "VGG13 with batch normalization": "vgg13_bn",
            "VGG16": "vgg16",
            "VGG16 with batch normalization": "vgg16_bn",
            "VGG19": "vgg19",
            "VGG19 with batch normalization": "vgg19_bn",
        },
    },
}


# Define possible optimizers in a dict.
# Format: optimizer -> default learning rate
OPTIMIZERS = {
    "Adam": 0.001,
    "Adadelta": 1.0,
    "Adagrad": 0.01,
    "Adamax": 0.002,
    "RMSprop": 0.01,
    "SGD": 0.1,
}


def show():
    """Shows the side bar and returns user inputs as dict."""

    inputs = {}

    with st.sidebar:
        st.write("## Task")
        inputs["task"] = st.selectbox(
            "Which type of problem do you want to solve?",
            ("Supervised Learning", "Unsupervised Learning", "Deep Learning"),
        )
        if inputs["task"] == "Supervised Learning":
            st.write("## Model")
            # framework = st.selectbox("Which framework?", ("scikit-learn", "PyTorch"))
            framework = st.selectbox("Which framework?", ("scikit-learn",))
            inputs["framework"] = framework
            model_task = inputs["task"]
            model_variant = st.selectbox("Which model type?", list(MODELS[framework][model_task].keys()))
            inputs["variant"] = model_variant
            model = st.selectbox("Which model?", list(MODELS[framework][model_task][model_variant].keys()))
            inputs["model"] = model
        elif inputs["task"] == "Unsupervised Learning":
            st.write("## Model")
            # framework = st.selectbox("Which framework?", ("scikit-learn", "PyTorch"))
            framework = st.selectbox("Which framework?", ("scikit-learn",))
            inputs["framework"] = framework
            model_task = inputs["task"]
            model_variant = st.selectbox("Which model type?", list(MODELS[framework][model_task].keys()))
            inputs["variant"] = model_variant
            model = st.selectbox("Which model?", list(MODELS[framework][model_task][model_variant].keys()))
            inputs["model"] = model
        elif inputs["task"] == "Deep Learning":
            st.write("## Model")
            # framework = st.selectbox("Which framework?", ("PyTorch", "scikit-learn"))
            framework = st.selectbox("Which framework?", ("PyTorch", ))
            inputs["framework"] = framework
            model = st.selectbox("Which model?", list(MODELS[framework].keys()))
            # Show model variants if model has multiple ones.
            if isinstance(MODELS[framework][model], dict):  # different model variants
                model_variant = st.selectbox(
                    "Which variant?", list(MODELS[framework][model].keys())
                )
                inputs["model_func"] = MODELS[framework][model][model_variant]
            else:  # only one variant
                inputs["model_func"] = MODELS[framework][model]

            if framework == "PyTorch":
                inputs["pretrained"] = st.checkbox("Use pre-trained model")
                if inputs["pretrained"]:
                    st.markdown(
                        '<sup>Pre-training on ImageNet with 1k classes, <a href="https://pytorch.org/docs/stable/torchvision/models.html">details</a></sup>',
                        unsafe_allow_html=True,
                    )
        else:
            # st.write(
            #     "Classify an image into one out of several classes, based on the image content (e.g. 'cat' or 'dog')."
            # )
            st.write("## Model")
            framework = st.selectbox("Which framework?", ("PyTorch", "scikit-learn"))
            inputs["framework"] = framework
            model = st.selectbox("Which model?", list(MODELS[framework].keys()))
            # Show model variants if model has multiple ones.
            if isinstance(MODELS[framework][model], dict):  # different model variants
                model_variant = st.selectbox(
                    "Which variant?", list(MODELS[framework][model].keys())
                )
                inputs["model_func"] = MODELS[framework][model][model_variant]
            else:  # only one variant
                inputs["model_func"] = MODELS[framework][model]

            if framework == "PyTorch":
                inputs["pretrained"] = st.checkbox("Use pre-trained model")
                if inputs["pretrained"]:
                    st.markdown(
                        '<sup>Pre-training on ImageNet with 1k classes, <a href="https://pytorch.org/docs/stable/torchvision/models.html">details</a></sup>',
                        unsafe_allow_html=True,
                    )

            st.write("## Input data")
            inputs["data_format"] = st.selectbox(
                "What best describes your input data?", ("Numpy arrays", "Image files")
            )
            if inputs["data_format"] == "Numpy arrays":
                st.write(
                    """
                Expected format: `[images, labels]`
                - `images` has array shape `(num samples, color channels, height, width)`
                - `labels` has array shape `(num samples, )`
                """
                )
            elif inputs["data_format"] == "Image files":
                st.write(
                    """
                Expected format: One folder per class, e.g.
                ```
                train
                +-- dogs
                |   +-- lassie.jpg
                |   +-- komissar-rex.png
                +-- cats
                |   +-- garfield.png
                |   +-- smelly-cat.png
                ```
                
                See also [this example dir](https://github.com/jrieke/traingenerator/tree/main/data/image-data)
                """
                )

            st.write("## Preprocessing")
            # st.checkbox("Convert to grayscale")
            # st.checkbox("Convert to RGB", True)
            if framework == "scikit-learn":
                if inputs["data_format"] == "Image files":
                    inputs["resize_pixels"] = st.number_input(
                        "Resize images to... (required for image files)", 1, None, 28
                    )
                    inputs["crop_pixels"] = st.number_input(
                        "Center-crop images to... (required for image files)",
                        1,
                        inputs["resize_pixels"],
                        min(28, inputs["resize_pixels"]),
                    )
                inputs["scale_mean_std"] = st.checkbox("Scale to mean 0, std 1", True)
            elif framework == "PyTorch":
                # TODO: Maybe show disabled checkbox here to make it more aligned with the
                #   display above.
                # st.markdown(
                #     '<label data-baseweb="checkbox" class="st-eb st-b4 st-ec st-d4 st-ed st-at st-as st-ee st-e5 st-av st-aw st-ay st-ax"><span role="checkbox" aria-checked="true" class="st-eg st-b2 st-bo st-eh st-ei st-ej st-ek st-el st-bb st-bj st-bk st-bl st-bm st-em st-en st-eo st-ep st-eq st-er st-es st-et st-av st-aw st-ax st-ay st-eu st-cb st-ev st-ew st-ex st-ey st-ez st-f0 st-f1 st-f2 st-c5 st-f3 st-f4 st-f5" style="background-color: rgb(150, 150, 150);"></span><input aria-checked="true" type="checkbox" class="st-b0 st-an st-cv st-bd st-di st-f6 st-cr" value=""><div class="st-ev st-f7 st-bp st-ae st-af st-ag st-f8 st-ai st-aj">sdf</div></label>',
                #     unsafe_allow_html=True,
                # )
                st.write("Resize images to 256 (required for this model)")
                st.write("Center-crop images to 224 (required for this model)")
                if inputs["pretrained"]:
                    st.write("Scale mean and std for pre-trained model")

            st.write("## Training")
            if framework == "scikit-learn":
                st.write("No additional parameters")
            elif framework == "PyTorch":
                inputs["gpu"] = st.checkbox("Use GPU if available", True)
                inputs["checkpoint"] = st.checkbox("Save model checkpoint each epoch")
                if inputs["checkpoint"]:
                    st.markdown(
                        "<sup>Checkpoints are saved to timestamped dir in `./checkpoints`. They may consume a lot of storage!</sup>",
                        unsafe_allow_html=True,
                    )
                inputs["loss"] = st.selectbox(
                    "Loss function", ("CrossEntropyLoss", "BCEWithLogitsLoss")
                )
                inputs["optimizer"] = st.selectbox("Optimizer", list(OPTIMIZERS.keys()))
                default_lr = OPTIMIZERS[inputs["optimizer"]]
                inputs["lr"] = st.number_input(
                    "Learning rate", 0.000, None, default_lr, format="%f"
                )
                inputs["batch_size"] = st.number_input("Batch size", 1, None, 128)
                inputs["num_epochs"] = st.number_input("Epochs", 1, None, 3)
                inputs["print_every"] = st.number_input(
                    "Print progress every ... batches", 1, None, 1
                )

            st.write("## Visualizations")
            inputs["visualization_tool"] = st.selectbox(
                "How to log metrics?", ("Not at all", "Tensorboard", "comet.ml")
            )
            if inputs["visualization_tool"] == "comet.ml":
                # TODO: Add a tracker how many people click on this link.
                "[Sign up for comet.ml](https://www.comet.ml/) :comet: "
                inputs["comet_api_key"] = st.text_input("Comet API key (required)")
                inputs["comet_project"] = st.text_input("Comet project name (optional)")
            elif inputs["visualization_tool"] == "Tensorboard":
                st.markdown(
                    "<sup>Logs are saved to timestamped dir in `./logs`. View by running: `tensorboard --logdir=./logs`</sup>",
                    unsafe_allow_html=True,
                )

        # "Which plots do you want to add?"
        # # TODO: Show some examples.
        # st.checkbox("Sample images", True)
        # st.checkbox("Confusion matrix", True)

        # "## Saving"
        # st.checkbox("Save config file", True)
        # st.checkbox("Save console output", True)
        # st.checkbox("Save finished model", True)
        # if model in TORCH_MODELS:
        #     st.checkbox("Save checkpoint after each epoch", True)

    return inputs
