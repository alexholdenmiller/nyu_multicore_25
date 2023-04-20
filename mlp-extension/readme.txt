# Multicore Computing: 
final project for group 25 of NYU's spring 2023 multicore class
{ahm9968, ajk745, jc12020, jp6862}@nyu.edu

# setup

```shell
scp nyu_multicore_25.zip xxx@access.cims.nyu.edu:~/
ssh xxx@access.cims.nyu.edu
ssh crunchy5.cims.nyu.edu
mkdir tmp
mv nyu_multicore_25.zip tmp
cd tmp
unzip nyu_multicore_25
module load python-3.9
virtualenv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ninja
module load gcc-9.2 
python setup.py install
python test.py
```