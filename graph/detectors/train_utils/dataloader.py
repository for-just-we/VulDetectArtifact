import random
from tap import Tap

class TrainParser(Tap):
    max_epochs: int = 100
    early_stopping: int = 5
    save_epoch: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1.3e-6

train_args = TrainParser().parse_args(known_only=True)

# 传入的每个sample应为Tuple[int, List[Data], torch.LongTensor]
def get_dataloader(positive_samples, negative_samples, batch_size: int):
    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)

    num_batch = len(all_samples) // batch_size
    if len(all_samples) % batch_size != 0:
        num_batch += 1

    for i in range(num_batch):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(all_samples))

        yield all_samples[start_idx: end_idx]