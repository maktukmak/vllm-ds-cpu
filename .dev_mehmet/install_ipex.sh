apt install python-is-python3

mkdir ipex_bundle
cd ipex_bundle
wget https://github.com/intel/intel-extension-for-pytorch/raw/main/scripts/compile_bundle.sh
wget https://github.com/conda-forge/miniforge/releases/download/24.7.1-2/Miniforge3-24.7.1-2-Linux-x86_64.sh
bash miniforge.sh -b -p ./miniforge3
/workspaces/vllm/ipex_bundle/miniforge3/bin/activate
export PATH="/workspaces/vllm/ipex_bundle/miniforge3/bin:$PATH"

bash compile_bundle.sh
