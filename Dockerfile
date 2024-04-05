FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

LABEL author="Kelynn Nowakowski"
LABEL description="Default container definition for course project. Based on https://parsertongue.org/tutorials/using-the-ua-hpc/#constructing-a-docker-image"

# This will be our default directory for subsequent commands
WORKDIR /app

# Installing pandas, scikit-learn, numpy, nltk, ignite, etc.
RUN conda install -y pandas scikit-learn numpy nltk ignite -c pytorch \
    && pip install -U tensorboardX crc32c soundfile

# SpaCy
RUN conda install -y spacy cupy -c conda-forge
#RUN python -m spacy download en_core_web_trf

# Installing huggingface transformers, tokenizers, and the datasets library
# Specific version of transformers and the latest versions of tokenizers and datasets compatible with that version
RUN pip install -U transformers==4.17.0 \
    && pip install -U tokenizers datasets

# Installing ipython as a better default REPL and jupyter for running notebooks
RUN conda install -y ipython jupyter ipywidgets widgetsnbextension \
    && jupyter nbextension enable --py widgetsnbextension

# Default command for this image: Print the version for PyTorch installation
CMD ["python", "-c", "import torch; print(torch.__version__)"]

# Copy executables to path
COPY . ./
RUN chmod u+x scripts/* \
    && mv scripts/* /usr/local/bin/ \
    && rmdir scripts

# Launch jupyter by default
CMD ["/bin/bash", "launch-notebook"]