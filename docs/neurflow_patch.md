# Steps to install Neurflow  
- Create: Dong-You (Sam) Jhong (2/17/2025)
- Mend: Dong-You (Sam) Jhong (2/18/2025)
- ADD 14: Dong-You (Sam) Jhong (2/19/2025)



1. To use conda in HiperGator, run this in terminal:   
    ``` bash
    module load conda
    ```    
2. In environment.yml, comment out all nvidia/torch related packages because HiperGator has cuda already (use `nvidia-smi` to check cuda version), and create env by using conda  
3. Activate conda env.   
4. Manually install torch related packages, make sure to match or below the version showing in `nvidia-smi`. For example, cuda 12.4:   
    ``` bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```   
5. Install `choldate` (see the repo instruction)   
6. Bypass circular import issue between `utils.py` and `NeurFlow.py`:  
   In `utils.py` line 14, comment out and copy `from NeurFlow import Framework`, paste it around line 450 before the `def visualize_chain_of_concepts` function      
7. Create `imagenet` folder    
8. Copy dataset `/data/ai/ref-data/image/ImageNet/imagenet1k/Data/CLS-LOC/val` to `imagenet`. For some reason, the 2012 val dataset downloaded from image-net.org doesn't have labels
9. Download `ILSVRC2012_devkit_t12.tar.gz` to `imagenet`:   
   ```bash
   wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
   ```    
10. Directories should be:    
    ```bash
    imagenet/
    │── ILSVRC2012_devkit_t12.tar.gz
    │── val
    ```         
11. Modify data root in run.sh, locate the `imagenet` folder, use `pwd` command to check where you are:    
    ```bash
    DATA_DIR="/blue/ruogu.fang/jhong.dongyou/neurflow/data/imagenet"
    ```
12. Execute `run.sh`:
    ```bash
    cd scripts;
    bash run.sh
    ```
13. Create a file named `.env.secret` under `notebooks` folder with the following content (**It's crucial that you create a .gitignore file not to push the .env file to your GitHub/GitLab and leak your API key!**): 
    ```bash
    OPENAI_API_KEY=your_key_here
    ```
    - Through terminal, you can use `nano` or `vim` command to edit it, and use `ls -la` to find the hidden files     

14. Try notebooks (On HiperGator, you need to link created conda env to jupyter kernal, see [here](https://help.rc.ufl.edu/doc/Managing_Python_environments_and_Jupyter_kernels)):
    - Replace `data_dir` to your `imagenet` path, like we did in `run.sh`
    - If you cannot import `utils`, add this at the top of notebook:
      ```python
      import sys
      sys.path.append('../')
      ```   
    - In `Visualizing_circuit.ipynb`, change the import order to solve circular import issue:      
        ```python
        from utils import visualize_chain_of_concepts
        from NeurFlow import Framework
        ```
    - In `Explaining_concept.ipynb`, if you face the exceeding token issue, modify this line to only use the first 19 images (I think 1500 per image) because we have 30000 tokens limit:
        ```python
        related_imgs = list(set([index for index in related_imgs]))[:19]
        ```   
      
    
    