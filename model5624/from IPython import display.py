from IPython import display
display.clear_output()

import ultralytics
import torch
ultralytics.checks()
print(torch.cuda.is_available())