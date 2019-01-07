import inf

gpu = 0
epoch = 15
test_noisy = '/home/aaron/set/movie' 
out_path = '/home/aaron/out/DeepXi/test_v1/c1.0' + '_e' + str(epoch) 
model_path = '/home/aaron/model/DeepXi'
opt = 'y'

inf.DeepXi(test_noisy, out_path, model_path, epoch, gpu, opt)
