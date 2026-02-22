from model.model import *
from tokenizer.tokenizer import ConfigurableTokenizer
from tokenizer.dataloader_config import load_tokenizer_dataloader_config
from tokenizer.dataloader import create_tokenizer_dataloader
from tokenizer.loader import load_tokenizer_config
from training.dataloader_config import load_training_dataloader_config
from training.dataloader import TrainingDataLoader
from training.training_config import load_training_config


#def main() -> None:
    #config = load_config("model/gpt2_config.json")
    #tok_config = load_tokenizer_config("tokenizer/tok_config.json")
    #tok = ConfigurableTokenizer(tok_config)
    #dl_config = load_dataloader_config("tokenizer/dataloader_config.json")
    #tok.train_from_dataset(dl_config)
    #gpt = ConfigurableGPT(config)
    #tok.train_from_file(["datasets/shake.txt"])
    #print(gpt)
    #print(tok.tokenizer)
    #print(ConfigurableTokenizer.eval_tokenizer_on_file([5, 2], tok.tokenizer))
    #print_training_dataloader_batch()

def main():
    tok_config = load_tokenizer_config("tokenizer/tok_config.json")
    tok = ConfigurableTokenizer(tok_config)

    dataloader_config = load_tokenizer_dataloader_config("tokenizer/dataloader_config.json")

    tok.train_from_dataset(dataloader_config)

    print(ConfigurableTokenizer.eval_tokenizer_on_file([5], tok.tokenizer))



def print_dataloader_batch(config_path: str = "dataloader_config.json") -> None:
    config = load_tokenizer_dataloader_config(config_path)
    dataloader = create_tokenizer_dataloader(config, batch_size=10, num_workers=0)
    for batch in dataloader:
        print(batch)
        break


def print_training_dataloader_batch(
    config_path: str = "training/dataloader_config.json",
    tokenizer_config_path: str = "tokenizer/tok_config.json",
    training_config_path: str = "training/training_config.json",
    batch_size: int = 1):

    tok_config = load_tokenizer_config(tokenizer_config_path)
    tok = ConfigurableTokenizer(tok_config)
    dl_config = load_tokenizer_dataloader_config("tokenizer/dataloader_config.json")
    tok.train_from_dataset(dl_config)
    #tok.train_from_file(["datasets/shake.txt"])
    config = load_training_dataloader_config(config_path)
    train_config = load_training_config(training_config_path)
    dataloader = TrainingDataLoader(
        config=config,
        tokenizer=tok.tokenizer,
        batch_size=batch_size,
        seq_len=train_config.seq_len,
        num_workers=0
    )
    inputs, targets = dataloader.next_batch()
    print(inputs)
    #print(targets)
    inputs_list = inputs.tolist()
    targets_list = targets.tolist()
    decoded_inputs = tok.tokenizer.decode_batch(
        inputs_list, skip_special_tokens=False
    )
    decoded_targets = tok.tokenizer.decode_batch(
        targets_list, skip_special_tokens=False
    )
    print(decoded_inputs)
    print(decoded_targets)


if __name__ == "__main__":
    main()
