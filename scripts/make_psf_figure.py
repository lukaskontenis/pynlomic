"""Plot PSF

If the trace does not reach zero due to background you can disable the zero
line plotting by setting show_y_zero to False.
"""

from lcmicro.report import gen_thg_psf_fig
import matplotlib.pyplot as plt

gen_thg_psf_fig(file_name='800_PSF.txt', wavl=0.8,
                suptitle_suffix='Second surface, LCM1, CRONUS VIS beam',
                show_y_zero_marker=False)

gen_thg_psf_fig(file_name='1050_PSF.txt', wavl=1.05,
                suptitle_suffix='Second surface, LCM1, CRONUS NIR beam')

plt.show()
