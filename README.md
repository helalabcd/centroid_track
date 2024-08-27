# helalab


Training the simplest form of a HeLa tracking network. This repository takes a single frame and predicts the centroid map (a 2d grid with gaussians around the centroid locations). During inference, the tracking graph DOES NOT include edges between the nodes.



When evaluated with evaluate_aogm, it can be observed that AOGM (the erorr metric / the distance from the tracking graph as described by the label) on the validation set decrease:
![image](https://github.com/user-attachments/assets/d1f5f404-d4bc-4f72-9ad1-57fb2df425b1)

(This is evaluated using only the first burst of validation bursts. X axis := epochs / 100, y axis:= aogm  

To reproduce:
```
// Run training
sbatch -q7d train.sbatch
// Evaluate the saved models in parallel
python run_multi_eval.py (this will schedule one slurm job per saved model to evaluate aogm)
```
