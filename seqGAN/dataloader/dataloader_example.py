from DailyDialogLoader import DailyDialogLoader
from DailyDialogLoader import PadCollate
from torch.utils.data import Dataset, DataLoader


PATH_TO_DATA = 'data/dailydialog/train/dialogues_train.txt'


dd_loader = DailyDialogLoader(PATH_TO_DATA)

# Play with the batch size if you want:
dataloader = DataLoader(dd_loader, batch_size=4, shuffle=True, num_workers=4, collate_fn=PadCollate())

for i, input_target_pair in enumerate(dataloader):
	print('\ninput:')
	print(input_target_pair[0])
	print('\ninput shape:', input_target_pair[0].shape)
	print('\ntarget:')
	print(input_target_pair[1])
	print('\ntarget shape:', input_target_pair[1].shape)
	print('\n')
	if  i == 0:
		break
