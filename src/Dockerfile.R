FROM rocker/r-ver:4.4.1

RUN mkdir code

WORKDIR /code

COPY data/ /code/data/

COPY r-version/install_packages.R /code/r-version/
RUN Rscript /code/r-version/install_packages.R

COPY r-version/ /code/r-version/

WORKDIR /code/r-version

CMD ["/bin/bash"]