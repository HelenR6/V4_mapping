import numpy as np
from scipy.stats.stats import pearsonr
import h5py

# seed_list=[42,23,27,88,66]
seed_list=[42,23,27,88]
session_list=[
              'm_ohp_session1',
              'm_ohp_session2',
              'm_stretch_session1',
              'm_stretch_session2',
            #   'n_stretch_session1',
              's_ohp_session1',
              's_stretch_session1'
              ]
model="st_resnet"
# input_path="/home/helenr6"
# with open(f'{input_path}/V4/{session}/{model}_natural_mean.json') as json_file:
#     layerlist=[]
#     load_data = json.load(json_file)
#     json_acceptable_string = load_data.replace("'", "\"")
#     d = json.loads(json_acceptable_string)
#     # get the layer with the highest ID neural prediction. 
#     self.best_layer=max(d, key=d.get)
#     layerlist.append(max_natural_layer)

for session in session_list:
    session_path=session.replace('_','/')
    final_path=session_path[:-1]+'_'+session_path[-1:]
    f = h5py.File('/home/helenr6/npc_v4_data.h5','r')
    n1 = f.get('neural/naturalistic/monkey_'+final_path)[:]
    natural_neuron_target=np.mean(n1, axis=0)
    n2=f.get('neural/synthetic/monkey_'+final_path)[:]
    synth_neuron_target=np.mean(n2, axis=0)
    cc=0
    total_natural_corr=[]
    total_synth_corr=[]
    for seed in seed_list:
        natural_prediction=np.load(f"/home/helenr6/V4_mapping/{session}/mapper_{model}_{seed}_natural_prediction.npy",allow_pickle=True)
        synth_prediction=np.load(f"/home/helenr6/V4_mapping/{session}/mapper_{model}_{seed}_synth_prediction.npy",allow_pickle=True)
        if cc==0:
            natural_corr_array= np.array([pearsonr(natural_prediction[:, i], natural_neuron_target[:, i])[0] for i in range(natural_prediction.shape[-1])])
            synth_corr_array= np.array([pearsonr(synth_prediction[:, i], synth_neuron_target[:, i])[0] for i in range(synth_prediction.shape[-1])])
            total_natural_corr=natural_corr_array
            total_synth_corr=synth_corr_array
            cc=cc+1
        else:
            natural_corr_array= np.array([pearsonr(natural_prediction[:, i], natural_neuron_target[:, i])[0] for i in range(natural_prediction.shape[-1])])
            synth_corr_array= np.array([pearsonr(synth_prediction[:, i], synth_neuron_target[:, i])[0] for i in range(synth_prediction.shape[-1])])
            total_natural_corr=np.vstack([total_natural_corr,natural_corr_array])
            total_synth_corr=np.vstack([total_synth_corr,synth_corr_array])
            cc=cc+1
    np.save(f'/home/helenr6/V4_mapping/{session}/mapper_{model}_synth_neuron_corr.npy',total_synth_corr)
    np.save(f'/home/helenr6/V4_mapping/{session}/mapper_{model}_natural_neuron_corr.npy',total_natural_corr)