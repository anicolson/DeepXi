import inf

gpu = 0
epoch = 15
snr = [-10, -5, 0, 5, 10, 15, 20]
test_clean = '/home/aaron/set/DeepXi/test_v1/test_clean'
test_noise = '/home/aaron/set/DeepXi/test_v1/test_noise' 
out_path = '/home/aaron/aaron/home/aaron/out/DeepXi/test_v1/DeepXi_3c30' + '_e' + str(epoch) 
model_path = '/home/aaron/model/DeepXi'

inf.DeepXi(test_clean, test_noise, out_path, model_path, epoch, snr, gpu)
