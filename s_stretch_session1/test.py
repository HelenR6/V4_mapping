import numpy as np
temp=np.load("st_resnet_42_synth_prediction.npy",allow_pickle=True)
print(temp.shape)
# session="s_stretch_session1"
# arch="st_resnet"
# seed=42
# _array=np.load(f'/home/helenr6/V4_mapping/{session}/{arch}_{seed}_synth_prediction.npy',allow_pickle=True)

# # _array[np.isnan(_array)] = 0
# # print(type(_array[4][0]))
# # sum_array=np.sum(_array,axis=0)
# print(np.isnan(_array))
# sum_array[np.isnan(sum_array)] = 0
# sum_array=sum_array/5
# np.save(f'/home/helenr6/V4_mapping/{session}/{arch}_{seed}_synth_prediction.npy',sum_array)