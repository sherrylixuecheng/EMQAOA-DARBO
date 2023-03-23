import os

for i in range(1,3):
	for k in range(5):
		for j in ['ideal', 'raw', 'mit']:
			os.system('mv Y_{}_hardware_exp_DARRBO_optmit_100_trial{}_p{}.npy Y_{}_hardware_exp_DARBO_optmit_100_trial{}_p{}.npy'.format(j,k,i,j,k,i))