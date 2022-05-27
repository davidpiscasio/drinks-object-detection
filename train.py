import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from train_arguments import get_args
from data_module import KWSDataModule
from pytorch_lightning import Trainer
from transformer_model import KWSTransformer

if __name__ == "__main__":
    args = get_args()
    CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
               'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    
    # make a dictionary from CLASSES to integers
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    datamodule = KWSDataModule(batch_size=args.batch_size,
                            patch_num=args.patch_num, 
                            num_workers=args.num_workers * args.devices,
                            path=args.path, n_fft=args.n_fft, n_mels=args.n_mels,
                            win_length=args.win_length, hop_length=args.hop_length,
                            class_dict=CLASS_TO_IDX)
    datamodule.setup()

    data = iter(datamodule.train_dataloader()).next()
    patch_dim = data[0].shape[-1]
    seqlen = data[0].shape[-2]
    print("Embed dim:", args.embed_dim)
    print("Patch size:", 32//args.patch_num)
    print("Sequence length:", seqlen)

    model = KWSTransformer(num_classes=args.num_classes, lr=args.lr, epochs=args.max_epochs, 
                           depth=args.depth, embed_dim=args.embed_dim, head=args.num_heads,
                           patch_dim=patch_dim, seqlen=seqlen)

    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.path, "checkpoints"),
        filename="transformer-kws-best-acc",
        save_top_k=1,
        verbose=True,
        monitor='test_acc',
        mode='max')

    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
    trainer = Trainer(accelerator=args.accelerator, devices=args.devices,
                      max_epochs=args.max_epochs, precision=16 if args.accelerator == 'gpu' else 32, 
                      callbacks=model_checkpoint)
    model.hparams.sample_rate = datamodule.sample_rate
    model.hparams.idx_to_class = idx_to_class
    trainer.fit(model, datamodule=datamodule)
    #trainer.test(model, datamodule=datamodule)

    script = model.to_torchscript()

    model = model.load_from_checkpoint(os.path.join(
    args.path, "checkpoints", "transformer-kws-best-acc.ckpt"))
    model.eval()
    script = model.to_torchscript()

    # save for use in production environment
    model_path = os.path.join(args.path, "checkpoints", 
                            "transformer-kws-best-acc.pt")
    torch.jit.save(script, model_path)    