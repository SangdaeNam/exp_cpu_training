# exp_cpu_training
Experiment code for cpu training with Openvino IR
* To use Intel internal graphics -> lspci | grep VGA
* If there is no Intel GPU, we have to enable IGPU on bios setting
```
BIOS window -> Advanced Mode -> System Agent Configuration -> Graphics Configuration -> IGPU Multi-Monitor -> (Disable->Enable) -> Save&Exit
```

* If there is Intel GPU(ex.Intel Corporation Device 4c8a)
  - sudo apt-get install clinfo
  - [Check intel.icd is installed](https://github.com/openvinotoolkit/openvino/wiki/GPUPluginDriverTroubleshooting)
  - If intel.icd is not installed -> Installation with ./install_NEO_OCL_driver.sh [following link](https://github.com/openvinotoolkit/openvino/blob/master/scripts/install_dependencies/install_NEO_OCL_driver.sh)

***
- If IGPU is ready, run experiments with following progress
```
git clone
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python transfer_hybrid_main.py
```
