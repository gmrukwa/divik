FROM divik:dev

EXPOSE 8888

RUN apt-get update \
    && apt-get install -y \
        gcc \
        git \
        ssh \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    --no-cache-dir \
    jupyterlab \
    line_profiler \
    memory_profiler

CMD ["jupyter", "lab", "--no-browser", "--allow-root", "--ip=0.0.0.0", "--NotebookApp.token=''"]
