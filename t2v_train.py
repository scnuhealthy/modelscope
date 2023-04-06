import pathlib
from modelscope.pipelines import pipeline
import math
from tqdm.auto import tqdm
from modelscope.models.base import Model
from typing import Dict, Optional, Tuple
from accelerate import Accelerator
import torch
from diffusers.optimization import get_scheduler
from einops import rearrange
import torch.nn.functional as F
from diffusers import DDIMScheduler
from torch.utils.data import Dataset
from diffusers.utils.import_utils import is_xformers_available
import decord
from accelerate.logging import get_logger
import numpy as np
import imageio
from modelscope.pipelines import pipeline
decord.bridge.set_bridge('torch')

logger = get_logger(__name__, log_level="INFO")

def _scale_timesteps(t):
    rescale_timesteps = False
    if rescale_timesteps:
        return t.float() * 1000.0 / self.num_timesteps
    return t

class VideoDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 16,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        self.video_path = video_path
        self.prompt = prompt

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "text": self.prompt
        }
        # video = example['pixel_values'].permute(0,2,3,1)
        # images = []
        # for image in video:
        #     image = image * 0.5 +0.5
        #     image.clamp_(0, 1)
        #     image = (image*255).numpy().astype(np.uint8)
        #     images.append(image)
        # imageio.mimsave('data_test.gif', images, fps=8)
        return example


def main():
    gradient_accumulation_steps = 1
    max_train_steps = 3000
    global_step = 0
    first_epoch = 0
    mixed_precision = "fp16"
    scale_lr = True
    train_batch_size = 1
    lr_scheduler = 'constant'
    learning_rate = 3e-4
    lr_warmup_steps = 0
    video_path = '../data/car-turn.mp4'
    prompt = 'a jeep car is moving on the road'
    max_grad_norm = 1.0
    enable_xformers_memory_efficient_attention = False
    checkpointing_steps = 500
    validation_steps = 250

    # # Get the traning dataset
    # train_dataset = VideoDataset(video_path=video_path, prompt=prompt)
    # train_dataset[0]

    # load models
    model_dir = pathlib.Path('weights')
    model = Model.from_pretrained(model_dir.as_posix(), model_prefetched=False)
    validation_pipeline = pipeline('text-to-video-synthesis',model=model)

    noise_scheduler = DDIMScheduler(num_train_timesteps=1000,beta_start=0.00085,beta_end=0.0120,beta_schedule='scaled_linear')
    text_encoder = model.clip_encoder
    vae = model.autoencoder
    unet = model.sd_model
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    trainable_modules: Tuple[str] = (
        "attn1.to_q",
        "attn2.to_q",
        "attn_temp",
    )
    unet.requires_grad_(False)
    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True

    # accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # xformer 
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # learning rate
    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # optimizer
    optimizer_cls = torch.optim.AdamW
    adam_beta1  = 0.9
    adam_beta2  = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-08
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Get the traning dataset
    train_dataset = VideoDataset(video_path=video_path, prompt=prompt)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    progress_bar = tqdm(range(global_step, max_train_steps),disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert videos to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                # print(pixel_values.shape, pixel_values.device,pixel_values.dtype)
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).sample()
                # print(latents.shape,latents.device,latents.dtype)
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215 # why

                # Sample noise that we'll add to the latent
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # print('1111', timesteps)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                # print('2222', noisy_latents.shape)
                # Get the target for loss depending on the prediction type
                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")


                text_emb = text_encoder(batch['text'])
                model_pred = unet(x=noisy_latents, t=_scale_timesteps(timesteps), y=text_emb)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # Create the pipeline using the trained modules and save it.
                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_unet = accelerator.unwrap_model(unet)
                        ckpt_name = 'new_unet_step_%d.pth' % global_step
                        torch.save(save_unet.state_dict(), ckpt_name)

                if global_step % validation_steps == 0:
                    if accelerator.is_main_process:
                        unet.eval()
                        output_video = 'step%d.mp4' % global_step 
                        validation_pipeline({'text':prompt},output_video=output_video)
                        unet.train()
        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

        if global_step >= max_train_steps:
            break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        torch.save(unet.state_dict(), 'new_unet.pth')
    accelerator.end_training()

if __name__ == '__main__':
    main()