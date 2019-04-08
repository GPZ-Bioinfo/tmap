FROM python:latest
RUN git clone https://github.com/GPZ-Bioinfo/tmap.git \
    && cd tmap \
    && python setup.py install \
    && pip install scikit-bio psutil \
    && apt update \
    && apt install r-base r-base-core r-base-dev libgconf-2-4 libgtk2.0-0 libnss3 libasound2 xvfb -y \
    && R -e "install.packages('vegan',repo='http://cran.rstudio.com/')" \
    && wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p \
    && export PATH="$HOME/miniconda3/bin:$PATH" \
    && conda install -c plotly plotly-orca -y \
    && touch /usr/bin/orca; echo "#!""/bin/bash" > /usr/bin/orca; echo "xvfb-run -a /root/miniconda3/bin/orca" "$""@" >> /usr/bin/orca; chmod +x /usr/bin/orca \
    && chmod a+rx /root; mkdir -m a+rwx /.config;  
ENV XDG_CONFIG_HOME=/.config/
