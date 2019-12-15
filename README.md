# Bad Global Minima Exist and SGD Can Reach Them

Here's the environment set-up:

```
wget -q –retry-connrefused –waitretry=10 https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh
chmod 777 *
./Anaconda2-4.3.1-Linux-x86_64.sh -b -p ./anaconda > /dev/null
chmod 777 *

conda install --yes pyyaml > /dev/null
conda install --yes HDF5 > /dev/null
conda install --yes h5py > /dev/null
conda install --yes -c rdonnelly libgpuarray > /dev/null
conda install --yes -c rdonnelly pygpu > /dev/null
conda install --yes pytorch=0.3.1 torchvision -c soumith > /dev/null
chmod 777 -R ./anaconda
```
