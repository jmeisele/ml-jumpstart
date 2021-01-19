"""
Author: Jason Eisele
Date: December 21, 2020
Scope: Shows the sidebar for the streamlit app and manages all user inputs.
Email: jeisele@shipt.com
"""

import streamlit as st
import model_config

MODELS = model_config.models
OPTIMIZERS = model_config.optimizers


def show():
    """Shows the side bar and returns user inputs as dict."""

    inputs = {}

    with st.sidebar:
        st.write("## Task")
        inputs["task"] = st.selectbox(
            "Which type of Machine Learning Experiment?",
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

        st.write("## Preprocessing")
        if framework == "scikit-learn":
            # if inputs["data_format"] == "Image files":
            #     inputs["resize_pixels"] = st.number_input(
            #         "Resize images to... (required for image files)", 1, None, 28
            #     )
            #     inputs["crop_pixels"] = st.number_input(
            #         "Center-crop images to... (required for image files)",
            #         1,
            #         inputs["resize_pixels"],
            #         min(28, inputs["resize_pixels"]),
            #     )
            inputs["scale_mean_std"] = st.checkbox("Scale to mean 0, std 1", True)

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
        st.write("## Metrics")
        inputs["metrics_tool"] = st.selectbox(
            "How to log training metrics?", ("Not at all", "MLFlow")
        )
        if inputs["metrics_tool"] == "MLFlow":
            "https://mlflow.ds.us-east-1.shipt.com/"
            # inputs["comet_api_key"] = st.text_input("Comet API key (required)")
            # inputs["comet_project"] = st.text_input("Comet project name (optional)")
        # elif inputs["metrics_tool"] == "Tensorboard":
        #     st.markdown(
        #         "<sup>Logs are saved to timestamped dir in `./logs`. View by running: `tensorboard --logdir=./logs`</sup>",
        #         unsafe_allow_html=True,
        #     )

        # else:
        #     # st.write(
        #     #     "Classify an image into one out of several classes, based on the image content (e.g. 'cat' or 'dog')."
        #     # )
        #     st.write("## Model")
        #     framework = st.selectbox("Which framework?", ("PyTorch", "scikit-learn"))
        #     inputs["framework"] = framework
        #     model = st.selectbox("Which model?", list(MODELS[framework].keys()))
        #     # Show model variants if model has multiple ones.
        #     if isinstance(MODELS[framework][model], dict):  # different model variants
        #         model_variant = st.selectbox(
        #             "Which variant?", list(MODELS[framework][model].keys())
        #         )
        #         inputs["model_func"] = MODELS[framework][model][model_variant]
        #     else:  # only one variant
        #         inputs["model_func"] = MODELS[framework][model]

        #     if framework == "PyTorch":
        #         inputs["pretrained"] = st.checkbox("Use pre-trained model")
        #         if inputs["pretrained"]:
        #             st.markdown(
        #                 '<sup>Pre-training on ImageNet with 1k classes, <a href="https://pytorch.org/docs/stable/torchvision/models.html">details</a></sup>',
        #                 unsafe_allow_html=True,
        #             )

        #     st.write("## Input data")
        #     inputs["data_format"] = st.selectbox(
        #         "What best describes your input data?", ("Numpy arrays", "Image files")
        #     )
        #     if inputs["data_format"] == "Numpy arrays":
        #         st.write(
        #             """
        #         Expected format: `[images, labels]`
        #         - `images` has array shape `(num samples, color channels, height, width)`
        #         - `labels` has array shape `(num samples, )`
        #         """
        #         )
        #     elif inputs["data_format"] == "Image files":
        #         st.write(
        #             """
        #         Expected format: One folder per class, e.g.
        #         ```
        #         train
        #         +-- dogs
        #         |   +-- lassie.jpg
        #         |   +-- komissar-rex.png
        #         +-- cats
        #         |   +-- garfield.png
        #         |   +-- smelly-cat.png
        #         ```
                
        #         See also [this example dir](https://github.com/jrieke/traingenerator/tree/main/data/image-data)
        #         """
        #         )

        #     st.write("## Preprocessing")
        #     # st.checkbox("Convert to grayscale")
        #     # st.checkbox("Convert to RGB", True)
        #     if framework == "scikit-learn":
        #         if inputs["data_format"] == "Image files":
        #             inputs["resize_pixels"] = st.number_input(
        #                 "Resize images to... (required for image files)", 1, None, 28
        #             )
        #             inputs["crop_pixels"] = st.number_input(
        #                 "Center-crop images to... (required for image files)",
        #                 1,
        #                 inputs["resize_pixels"],
        #                 min(28, inputs["resize_pixels"]),
        #             )
        #         inputs["scale_mean_std"] = st.checkbox("Scale to mean 0, std 1", True)
        #     elif framework == "PyTorch":
        #         # TODO: Maybe show disabled checkbox here to make it more aligned with the
        #         #   display above.
        #         # st.markdown(
        #         #     '<label data-baseweb="checkbox" class="st-eb st-b4 st-ec st-d4 st-ed st-at st-as st-ee st-e5 st-av st-aw st-ay st-ax"><span role="checkbox" aria-checked="true" class="st-eg st-b2 st-bo st-eh st-ei st-ej st-ek st-el st-bb st-bj st-bk st-bl st-bm st-em st-en st-eo st-ep st-eq st-er st-es st-et st-av st-aw st-ax st-ay st-eu st-cb st-ev st-ew st-ex st-ey st-ez st-f0 st-f1 st-f2 st-c5 st-f3 st-f4 st-f5" style="background-color: rgb(150, 150, 150);"></span><input aria-checked="true" type="checkbox" class="st-b0 st-an st-cv st-bd st-di st-f6 st-cr" value=""><div class="st-ev st-f7 st-bp st-ae st-af st-ag st-f8 st-ai st-aj">sdf</div></label>',
        #         #     unsafe_allow_html=True,
        #         # )
        #         st.write("Resize images to 256 (required for this model)")
        #         st.write("Center-crop images to 224 (required for this model)")
        #         if inputs["pretrained"]:
        #             st.write("Scale mean and std for pre-trained model")

        #     st.write("## Training")
        #     if framework == "scikit-learn":
        #         st.write("No additional parameters")
        #     elif framework == "PyTorch":
        #         inputs["gpu"] = st.checkbox("Use GPU if available", True)
        #         inputs["checkpoint"] = st.checkbox("Save model checkpoint each epoch")
        #         if inputs["checkpoint"]:
        #             st.markdown(
        #                 "<sup>Checkpoints are saved to timestamped dir in `./checkpoints`. They may consume a lot of storage!</sup>",
        #                 unsafe_allow_html=True,
        #             )
        #         inputs["loss"] = st.selectbox(
        #             "Loss function", ("CrossEntropyLoss", "BCEWithLogitsLoss")
        #         )
        #         inputs["optimizer"] = st.selectbox("Optimizer", list(OPTIMIZERS.keys()))
        #         default_lr = OPTIMIZERS[inputs["optimizer"]]
        #         inputs["lr"] = st.number_input(
        #             "Learning rate", 0.000, None, default_lr, format="%f"
        #         )
        #         inputs["batch_size"] = st.number_input("Batch size", 1, None, 128)
        #         inputs["num_epochs"] = st.number_input("Epochs", 1, None, 3)
        #         inputs["print_every"] = st.number_input(
        #             "Print progress every ... batches", 1, None, 1
        #         )

        #     st.write("## Visualizations")
        #     inputs["visualization_tool"] = st.selectbox(
        #         "How to log metrics?", ("Not at all", "Tensorboard", "comet.ml")
        #     )
        #     if inputs["visualization_tool"] == "comet.ml":
        #         # TODO: Add a tracker how many people click on this link.
        #         "[Sign up for comet.ml](https://www.comet.ml/) :comet: "
        #         inputs["comet_api_key"] = st.text_input("Comet API key (required)")
        #         inputs["comet_project"] = st.text_input("Comet project name (optional)")
        #     elif inputs["visualization_tool"] == "Tensorboard":
        #         st.markdown(
        #             "<sup>Logs are saved to timestamped dir in `./logs`. View by running: `tensorboard --logdir=./logs`</sup>",
        #             unsafe_allow_html=True,
        #         )

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
