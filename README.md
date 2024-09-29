# helalab


Training the simplest form of a HeLa tracking network. This repository takes a single frame and predicts the centroid map (a 2d grid with gaussians around the centroid locations). During inference, the predicted tracking graph DOES NOT include edges between the nodes.



When evaluated with evaluate_aogm, it can be observed that AOGM (the erorr metric / the distance from the tracking graph as described by the label) on the validation set decreases:
![image](https://github.com/user-attachments/assets/d1f5f404-d4bc-4f72-9ad1-57fb2df425b1)

(This is evaluated using only the first burst of validation bursts. X axis := epochs / 100, y axis:= aogm  

To reproduce:
```
// Run training
sbatch -q7d train.sbatch
// Evaluate the saved models in parallel
python run_multi_eval.py (this will schedule one slurm job per saved model to evaluate aogm)
```

After `run_milti_eval.py` is done, cd into models and get the aogm scores with `cat $(ls *.txt | sort -V)`


## Results

### v1.0.0
It can be verified that learning the centroids take place:
![image](https://github.com/user-attachments/assets/dc56484b-8551-47b0-b39e-81ffc926b39b)

Average AOGM on the validation set decreases, until it hits a limit at ~500.
![image](https://github.com/user-attachments/assets/dc6f35c0-cc8b-43ee-9bd2-cd1362df5774)

Hypothesize this is due to the labels being too large, making the downstream evaluation task unable to tell them apart properly (they bleed into each other, the sharp borders between them in y are not learned in y_pred)
-> This could maybe be fixed with smaller centroids

Verify that the downstream task indeed has problems not with detecting the presence of cells, but rather with telling them apart properly. Notice how the top left corner is clear of any cells. Earlier during training, the model wrongly detects cells here. After 1400 epochs, the model is pretty clear about the fact that there are no cells in this area. The main issue seems to be telling the predicted cells apart.
![image](https://github.com/user-attachments/assets/525d738e-625c-48eb-af96-324669c3f712)

**Could detection (=> tracking) performance (as measured by aogm) be further improved with smaller centroids?**
