from lcmicro.report import gen_img_report


file_name = '18_01_04_.629_.dat'
rng = [0, 22000]
gamma = 1

gen_img_report(file_name=file_name, corr_fi=False, rng=rng, gamma=gamma, chan_ind=3)
