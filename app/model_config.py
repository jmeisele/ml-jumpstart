"""
Author: Jason Eisele
Date: January 19, 2021
Scope: Model object(s) configuration dictionary
"""

# Define possible models in a dict.
# Format of the dict:
# option 1: framework -> model -> code
# option 2 – if model has multiple variants: framework -> model -> model variant -> code
# option 3: framework -> problem -> model -> code

models = {
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
optimizers = {
    "Adam": 0.001,
    "Adadelta": 1.0,
    "Adagrad": 0.01,
    "Adamax": 0.002,
    "RMSprop": 0.01,
    "SGD": 0.1,
}
